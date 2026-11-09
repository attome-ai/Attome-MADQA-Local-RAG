"""Document deep read (T35): for list answers, read every matching page of the answer's documents.

    python -m lab.docread answers.jsonl experiments/thin_ocr_llm_select.json --model NAME --split dev [--ids set.json]

Half the hard failures are lists spread over many pages of one document ("Humans of HUD" has one profile on each
of pages 10, 14, 17, 20, 23, 25, 28 and 31), but the reader sees 9 pages: the top 3 and their neighbours. This
searches inside the documents the answer cites (plus the top-ranked one) for every page matching the question,
reads those pages in batches of 9 with an extract-only prompt, and merges the items. It stops when a batch adds
nothing. Non-list answers are left alone. Output ends in _docread.jsonl, for lab.score_pick or as a final step.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import looks_unanswered
from .answer import build_content, saved_rankings
from .complete import is_list_question, page_count
from .profiling import format_table, reset, snapshot, timed
from .read import call_server, final_json
from .run import read_jsonl, select_questions, sha256
from .score_pick import page_blocks, yes_probability
from .stages import PageIndex

EXTRACT = """Question: {question}

The pages above come from {where}. List every item on these pages that answers the question, exactly as
written, and give the page number of each. Include only items these pages actually show; if none do, return an
empty list. Don't repeat items already found: {found}
Reply with JSON only: {{"items": [{{"item": "...", "page": 1}}]}}"""


CLEAN = """Question: {question}

A candidate list, gathered page by page from a document: {items}

Each item was read from a page that answers the question, so keep items by default. Remove an item only if it is
clearly a heading, label, total or category name rather than an answer. Reduce each item to just the answer the
question asks for (for "names", just the name: "HUMANS OF HUD - Traci, Long Beach, CA" becomes "Traci"). Merge
duplicates and different spellings of the same item into one. Split an entry that holds two items ("Laura and
Jaime" becomes "Laura" and "Jaime"). Don't add anything that isn't in the candidate list.
Reply with JSON only: {{"answer": ["..."]}}"""


# A deep read gathers every matching item, which only suits questions asking for a set of things. Comparisons
# ("years with lower fees than X", dev/15) and single facts ("on what date", dev/117) gained wrong items.
COMPARE = re.compile(
    r"\b(lower|higher|more|less|greater|fewer|larger|smaller|than|exceed\w*|above|below|increase\w*|decrease\w*)\b", re.I
)
SINGLE = re.compile(
    r"^\s*(on\s+)?(what|which)\s+(date|day|year|month|time|amount|number|percentage|value)\b|^\s*(when|how\s+(many|much|long|old))\b",
    re.I,
)


FIND = """Question: {question}

The page above comes from {where}. If this page states the answer to the question, give it exactly as written,
short. If it doesn't, return an empty list.
Reply with JSON only: {{"answer": ["..."]}}"""

# Exact identifiers a document must contain to be about the same thing: case numbers like 20-cv-7311, codes
# mixing letters and digits. train/353 took 8 juror numbers from another case's document.
IDENTIFIER = re.compile(r"\b(?=[\w-]*\d)(?=[\w-]*[a-z-])[a-z0-9]+(?:-[a-z0-9]+)+\b|\b(?=\w*\d)(?=\w*[a-z])\w{5,}\b", re.I)


def identifiers(question: str) -> list[str]:
    """Case numbers and codes; not year ranges (2014-2019), "covid-19" or "3-5" (at least 3 digits)."""
    found = [m.group(0).lower() for m in IDENTIFIER.finditer(question)]
    return [f for f in found if sum(c.isdigit() for c in f) >= 3 and not re.fullmatch(r"\d{4}-\d{2,4}", f)]


def load_texts(pages_path: Path) -> dict[str, str]:
    texts: dict[str, list[str]] = {}
    with pages_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            texts.setdefault(row["document"], []).append(row.get("text", ""))
    return {d: " ".join(t).lower() for d, t in texts.items()}


ENTITY_STOP = set(
    "what which who whom whose when where how why does did was were are the this that these those and for from with about "
    "according report document page pages total number name names list years year state states city county company inc llc corp".split()
)


def entity_words(question: str) -> list[str]:
    """Capitalised words that name the person, place or organisation the question is about."""
    words = [t.strip(".'’-").lower() for t in re.findall(r"[A-Z][a-zA-Z'’.-]{2,}", question)]
    return [w for w in words if len(w) >= 4 and w not in ENTITY_STOP]


def entity_mismatch(question: str, cited: list[str], top: str, texts: dict[str, str]) -> bool:
    """The answer cites only documents that never mention the entity the question names, while the top-ranked
    document does (dev/36: Colin Allred's spouse income given for Eddie J. Mauro, whose own page was not shown)."""
    words = entity_words(question)
    if not words or not cited:
        return False
    mentions = lambda document: any(w in texts.get(document, "") for w in words)  # noqa: E731
    return not any(mentions(d) for d in cited) and mentions(top)


def same_subject(question: str, documents: list[str], texts: dict[str, str]) -> list[str]:
    """Keep documents containing every exact identifier the question names; all of them if none does."""
    ids = identifiers(question)
    if not ids:
        return documents
    kept = [d for d in documents if all(i in texts.get(d, "") for i in ids)]
    return kept or documents


GROUND = """Question: {question}
Proposed item: {item}

Find, on the page above, the exact words that show this item meets EVERY condition in the question (the kind of
thing asked for, and any time period, status, place or other condition). Copy those words exactly as printed.
If the page does not show it meeting all the conditions, leave the quote empty.
Reply with JSON only: {{"quote": "..."}}"""


def grounded(quote: str, page_text: str) -> bool:
    """The quote has to be real: most of its words appear in the page's text layer."""
    words = re.findall(r"[a-z0-9]+", quote.lower())
    if len(words) < 3:
        return False
    page_words = set(re.findall(r"[a-z0-9]+", page_text.lower()))
    return sum(w in page_words for w in words) / len(words) >= 0.8


CHECK_ITEM = """Question: {question}
Proposed item: {item}

According to the page above, is the proposed item one of the correct answers to exactly what the question asks
(the right person, product, case or category, not just something mentioned on the page)? Reply with one word: yes or no."""

# Qualifier-aware version: the plain check says yes to anything of the right kind, so it let in wrong extras
# for questions with a condition ("favorites", "former officers", "reported changes", "term ending").
CHECK_QUALIFIED = """Question: {question}
Proposed item: {item}

First find every condition in the question besides the kind of thing asked for: a time period, a status or role,
a place, a source, a "who/which did X". Does the page above show that the proposed item meets ALL of those
conditions, not only that it is the right kind of thing? If the page does not show it meeting a condition, answer no.
Reply with one word: yes or no."""


def gave_up(answer: list) -> bool:
    """looks_unanswered misses a bare "..." (dev/36)."""
    return looks_unanswered(answer) or not any(re.search(r"\w", str(a)) for a in answer)


ALLOW_COMPARE = False  # set by --allow-compare: comparison questions ("costing more than $13") are list questions too


def wants_set(question: str, answer: list) -> bool:
    if not is_list_question(question, answer) or SINGLE.search(question):
        return False
    return ALLOW_COMPARE or not COMPARE.search(question)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def parse_items(text: str) -> list[tuple[str, int]]:
    body = text.rsplit("</think>", 1)[-1]
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for entry in items if isinstance(items, list) else []:
        if isinstance(entry, dict) and str(entry.get("item", "")).strip():
            page = entry.get("page")
            out.append((str(entry["item"]).strip(), page if isinstance(page, int) else 0))
    return out


def matching_pages(
    index: PageIndex, query: str, documents: list[str], limit: int, counts=None, whole: int = 0
) -> list[tuple[str, int]]:
    """Pages of these documents that match the query, in reading order per document.

    A document of at most `whole` pages is read in full: search misses pages whose text layer lacks the
    words, such as a title set in an image (dev/56 missed 4 of 8 profile pages)."""
    wanted = set(documents)
    hits: list[tuple[str, int]] = []
    for document in documents:
        if counts and whole and counts(document) <= whole:
            hits += [(document, page) for page in range(1, counts(document) + 1)]
            wanted.discard(document)
    hits += [p for p in index.bm25(query, 6000) if p[0] in wanted]
    order = {d: n for n, d in enumerate(documents)}
    return sorted(hits[:limit], key=lambda p: (order[p[0]], p[1]))


def merge(answer: list[str], new: list[str]) -> list[str]:
    seen = {norm(a) for a in answer}
    out = list(answer)
    for item in new:
        if norm(item) and norm(item) not in seen:
            seen.add(norm(item))
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--max-pages", type=int, default=40, help="most pages to read per question")
    parser.add_argument("--whole", type=int, default=40, help="read documents up to this many pages in full")
    parser.add_argument("--batch", type=int, default=1, help="pages per request; with several pages the reader tends to list one item per batch")
    parser.add_argument("--no-clean", action="store_true", help="skip the precision pass on the merged list")
    parser.add_argument("--early-stop", action="store_true", help="stop when a batch adds no new item")
    parser.add_argument("--check", type=float, default=0.1, help="drop added items whose P(yes) on their pages is below this (0 = off); 0.5 dropped real items (RFeye View 0.13, an appendix 0.47)")
    parser.add_argument("--qualified", action="store_true", help="check added items against every condition in the question")
    parser.add_argument("--allow-compare", action="store_true", help="also deep-read list questions with a comparison")
    parser.add_argument("--scan", type=int, default=0, help="also read this many top-ranked pages one by one, across documents")
    parser.add_argument("--collect-only", action="store_true", help="deep-read only questions the model flags as needing every matching item (lab.spread)")
    parser.add_argument("--agree", action="store_true", help="keep an added item only if a second pass (pages reversed, batches of 2) also finds it")
    parser.add_argument("--ground", action="store_true", help="keep an added item only if the model quotes words from its page that show it meets every condition, and the quote is really on the page")
    parser.add_argument("--entity", action="store_true", help="treat an answer from documents that never mention the named entity as unanswered, and look for it in the top document")
    parser.add_argument("--only-find", action="store_true", help="skip the list deep read; only run the find step")
    parser.add_argument("--find", action="store_true",help="for unanswered questions, read the top document page by page")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    global ALLOW_COMPARE
    ALLOW_COMPARE = args.allow_compare
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    index = PageIndex(args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite")
    index.pages_path = pages_path
    rows = select_questions(read_jsonl(args.answers), args.ids)
    def call(content: list[dict]) -> dict:
        # Docker's port forward sometimes drops a connection (WinError 10054); one retry, then give up.
        try:
            return call_server(args.base_url, args.model, content)
        except (urllib.error.URLError, OSError):
            time.sleep(2)
            return call_server(args.base_url, args.model, content)
    clean = not args.no_clean
    texts = load_texts(pages_path)
    page_texts = {}
    if args.ground:
        from .hybrid import load_text

        page_texts = load_text(pages_path)
    early_stop = args.early_stop

    counts = lambda document: page_count(args.data / "pdfs" / document)
    collect = {}
    if args.scan or args.collect_only:
        from .spread import classify

        collect = classify([r["question"] for r in rows], args.model, args.base_url, cache=Path("data/cache/collect.json"))

    def find(row: dict, result: dict) -> dict:
        """The reader gave up. If the top-ranked document is the right one, the answer is often on a page that
        wasn't shown (dev/36: the document ranks 1st, its page 48th). Read it page by page; stop at the first
        page that answers."""
        documents = same_subject(row["question"], [ranked[row["id"]][0][0]], texts)
        pages = matching_pages(index, row["question"], documents, args.max_pages, counts, args.whole)
        for number, page in enumerate(pages, 1):
            content = build_content(row["question"], [page], args.data, "", args.max_side, True)
            content[-1] = {"type": "text", "text": FIND.format(question=row["question"], where=f"document {page[0]}")}
            with timed("docread_find"):
                try:
                    data = final_json(call(content)["choices"][0]["message"]["content"] or "")[0]
                except (urllib.error.URLError, OSError):
                    break
            answer = [str(a) for a in (data or {}).get("answer", []) if str(a).strip()] if isinstance(data, dict) else []
            if answer and not gave_up(answer):
                result.update(answer=answer, citations=[{"file": page[0], "page": page[1]}], docread=True,
                              iterations=int(row.get("iterations", 1)) + 1)
                break
        result.update(docread_pages=len(pages), docread_batches=number if pages else 0, docread_mode="find")
        return result

    def check_new_items(row: dict, answer: list[str], sources: dict, result: dict) -> list[str]:
        """Check each item the deep read added against the page it came from; drop the ones the model doubts.
        Items the first reader gave, or whose page is unknown, stay."""
        kept, dropped = [], []
        original = {norm(a) for a in row["answer"]} if not result.get("rebuilt") else set()
        for item in answer:
            key = norm(item)
            pages = sources.get(key) or [p for k, ps in sources.items() if key and key in k for p in ps]
            if key in original or not pages:
                kept.append(item)
                continue
            # An item is often first seen where it's only mentioned ("works with RFeye View"); its own page may
            # come later. Keep it if any of its pages (up to 3) confirms it.
            documents = {d for d, _ in pages}
            own = [p for p in index.bm25(str(item), 2000) if p[0] in documents][:2]
            best = 0.0
            for page in list(dict.fromkeys(pages[:1] + own))[:3]:
                content = page_blocks([page], args.data, args.max_side)
                content.append({"type": "text", "text": (CHECK_QUALIFIED if args.qualified else CHECK_ITEM).format(question=row["question"], item=json.dumps(item, ensure_ascii=False))})
                with timed("docread_check"):
                    try:
                        best = max(best, yes_probability(args.base_url, args.model, content))
                    except OSError:
                        best = 1.0
                if best >= args.check:
                    break
            if args.ground and best >= args.check:
                ok = False
                for page in list(dict.fromkeys(pages[:1] + own))[:3]:
                    content = page_blocks([page], args.data, args.max_side)
                    content.append({"type": "text", "text": GROUND.format(question=row["question"], item=json.dumps(item, ensure_ascii=False))})
                    with timed("docread_ground"):
                        try:
                            reply = call(content)["choices"][0]["message"]["content"] or ""
                        except (urllib.error.URLError, OSError):
                            ok = True
                            break
                    body = reply.rsplit("</think>", 1)[-1]
                    match = re.search(r'"quote"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
                    if match and grounded(match.group(1), page_texts.get(page, "")):
                        ok = True
                        break
                result.setdefault("ground", []).append([item, ok])
                if not ok:
                    best = 0.0
            (kept if best >= args.check else dropped).append(item)
            result.setdefault("check_scores", []).append([item, round(best, 3)])
        result["dropped"] = dropped
        return kept or answer

    def one(row: dict) -> dict:
        result = {**row, "docread_answer": row["answer"], "docread": False}
        mismatch = args.entity and entity_mismatch(
            row["question"], list(dict.fromkeys(c["file"] for c in row.get("citations", []))), ranked[row["id"]][0][0], texts
        )
        result["entity_mismatch"] = bool(mismatch)
        if gave_up(row["answer"]) or mismatch:
            return find(row, result) if args.find else result
        if args.only_find:
            return result
        flagged = collect.get(row["question"], False)
        if args.collect_only:
            if not flagged:
                return result
        elif not (wants_set(row["question"], row["answer"]) or (args.scan and flagged)):
            return result
        cited = list(dict.fromkeys(c["file"] for c in row.get("citations", [])))
        documents = same_subject(row["question"], list(dict.fromkeys(cited + [ranked[row["id"]][0][0]])), texts)
        # The answer leans on a document about something else (another case number): rebuild it from the
        # right documents only, rather than adding to it.
        rebuild = any(d not in documents for d in cited)
        query = " ".join([row["question"]] + ([] if rebuild else [str(a) for a in row["answer"][:3]]))
        pages = matching_pages(index, query, documents, args.max_pages, counts, args.whole)
        if args.scan and flagged:
            pages = list(dict.fromkeys(pages + ranked[row["id"]][: args.scan]))
        answer = [] if rebuild else list(row["answer"])
        citations = [] if rebuild else list(row.get("citations", []))
        batches = 0
        sources: dict[str, tuple[str, int]] = {}
        result["rebuilt"] = rebuild
        for start in range(0, len(pages), args.batch):
            batch = pages[start : start + args.batch]
            where = "document " + ", ".join(dict.fromkeys(d for d, _ in batch))
            content = build_content(row["question"], batch, args.data, "", args.max_side, True)
            prompt = EXTRACT.format(question=row["question"], where=where, found=json.dumps(answer, ensure_ascii=False))
            content[-1] = {"type": "text", "text": prompt}
            with timed("docread_batch"):
                try:
                    text = call(content)["choices"][0]["message"]["content"] or ""
                except (urllib.error.URLError, OSError) as error:
                    print(f"REQUEST FAILED {row['id']}: {error}", flush=True)
                    break
            batches += 1
            found = parse_items(text)
            for item, _ in found:
                sources.setdefault(norm(item), []).append(batch[0])
            before = len(answer)
            answer = merge(answer, [item for item, _ in found])
            for _, page in found:
                if 1 <= page <= len(batch) and {"file": batch[page - 1][0], "page": batch[page - 1][1]} not in citations:
                    citations.append({"file": batch[page - 1][0], "page": batch[page - 1][1]})
            if early_stop and len(answer) == before and start > 0:
                break
        result.update(docread_pages=len(pages), docread_batches=batches)
        if args.agree and answer != row["answer"]:
            second: set[str] = set()
            backwards = list(reversed(pages))
            for start in range(0, len(backwards), 2):
                batch = backwards[start : start + 2]
                where = "document " + ", ".join(dict.fromkeys(d for d, _ in batch))
                content = build_content(row["question"], batch, args.data, "", args.max_side, True)
                content[-1] = {"type": "text", "text": EXTRACT.format(question=row["question"], where=where, found=json.dumps(list(row["answer"]), ensure_ascii=False))}
                with timed("docread_agree"):
                    try:
                        second |= {norm(i) for i, _ in parse_items(call(content)["choices"][0]["message"]["content"] or "")}
                    except (urllib.error.URLError, OSError):
                        break
            base = {norm(a) for a in row["answer"]}
            disagreed = [a for a in answer if norm(a) not in base and norm(a) not in second]
            answer = [a for a in answer if norm(a) in base or norm(a) in second]
            result["disagreed"] = disagreed
        if not answer:
            return result
        if clean and answer != row["answer"]:
            # Tidy pass, text only: each item already came from a page that was read. Showing pages here would
            # drop every item whose page didn't fit, as a list can span 30+ pages.
            content = [{"type": "text", "text": CLEAN.format(question=row["question"], items=json.dumps(answer, ensure_ascii=False))}]
            with timed("docread_clean"):
                try:
                    data = final_json(call(content)["choices"][0]["message"]["content"] or "")[0]
                except (urllib.error.URLError, OSError):
                    data = None
            if data and isinstance(data.get("answer"), list) and data["answer"]:
                answer = [str(a) for a in data["answer"] if str(a).strip()]
            result["cleaned"] = True
        if args.check and answer != row["answer"]:
            answer = check_new_items(row, answer, sources, result)
        if answer != row["answer"]:
            result.update(answer=answer, citations=citations, docread=True, iterations=int(row.get("iterations", 1)) + 1)
        return result

    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, rows))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.answers.with_name(f"{args.answers.stem}{suffix}_docread{'' if clean else '_raw'}.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    tried = sum("docread_batches" in o for o in outputs)
    print(f"List answers deep-read {tried}, changed {sum(o['docread'] for o in outputs)} of {len(outputs)}; "
          f"batches {sum(o.get('docread_batches', 0) for o in outputs)}; seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
