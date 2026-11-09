"""Sweep (T33): check every document of a kind for questions that ask across the collection.

    python -m lab.sweep drafts.jsonl experiments/thin_ocr_llm_select.json --model NAME --split dev

"Which restaurants list calories on their menus?" needs every menu checked, but search returns only the few
most similar pages. The model first says whether the question asks across many documents and what kind of
document to look in. If so, BM25 finds up to --docs documents of that kind; each is read on its 2 best pages
with a narrow yes/no-with-evidence prompt; the matches become one answer. Other questions keep the draft.
Output is a candidate for lab.adjudicate, ending in _sweep.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import build_content
from .profiling import format_table, reset, snapshot, timed
from .read import call_server
from .run import read_jsonl, select_questions, sha256
from .stages import PageIndex

DECIDE = """Question: {question}

Does answering this need checking many documents of the same kind and collecting every one that fits (for
example "which restaurants...", "whose cases...", "the earliest ... by any ...")? Or does it ask about one named
document, person or item? If it needs many documents, give a short keyword search that finds documents of that
kind (not the condition), and a short description of the condition each one must meet.
Reply with JSON only: {{"many": true, "kind_search": "...", "condition": "..."}}"""

CHECK = """Question: {question}
Condition to check in this document: {condition}

Look only at the pages above, which come from one document. Does this document meet the condition? If yes, give
the item this document contributes to the answer (a name, value or date, as the question asks), and quote the
supporting text. If no, or if the pages don't show it, say no.
Reply with JSON only: {{"answer": ["..."], "meets": true, "quote": "..."}}"""


def parse_json(text: str) -> dict | None:
    body = text.rsplit("</think>", 1)[-1]
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def documents_for(index: PageIndex, kind: str, question: str, limit: int) -> list[str]:
    """Documents matching the kind first, then the question, keeping first-seen order."""
    docs: list[str] = []
    for query in (kind, question):
        for document, _ in index.bm25(query, 400):
            if document not in docs:
                docs.append(document)
            if len(docs) >= limit:
                return docs
    return docs


def best_pages(index: PageIndex, question: str, condition: str, document: str, count: int) -> list:
    pages = [p for p in index.bm25(f"{question} {condition}", 4000) if p[0] == document][:count]
    return pages or [(document, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--docs", type=int, default=10)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    index = PageIndex(args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite")
    index.pages_path = pages_path
    drafts = select_questions(read_jsonl(args.drafts), args.ids)
    call = lambda content: call_server(args.base_url, args.model, content)
    text = lambda prompt: call([{"type": "text", "text": prompt}])["choices"][0]["message"]["content"] or ""

    def check(question: str, condition: str, document: str) -> tuple[list[str], list[dict]]:
        pages = best_pages(index, question, condition, document, args.pages)
        content = build_content(question, pages, args.data, "", args.max_side, True)
        content[-1] = {"type": "text", "text": CHECK.format(question=question, condition=condition)}
        try:
            data = parse_json(call(content)["choices"][0]["message"]["content"] or "")
        except (urllib.error.URLError, OSError):
            return [], []
        if not data or data.get("meets") is not True:
            return [], []
        answer = data.get("answer", [])
        answer = [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
        return [a for a in answer if a.strip()], [{"file": d, "page": p} for d, p in pages[:1]]

    def one(draft: dict) -> dict:
        result = {**draft, "swept": False}
        with timed("sweep_decide"):
            try:
                decision = parse_json(text(DECIDE.format(question=draft["question"])))
            except (urllib.error.URLError, OSError):
                return result
        if not decision or decision.get("many") is not True or not decision.get("kind_search"):
            return result
        condition = str(decision.get("condition", draft["question"]))
        documents = documents_for(index, str(decision["kind_search"]), draft["question"], args.docs)
        answer, citations = [], []
        with timed("sweep_check"):
            for document in documents:
                items, cited = check(draft["question"], condition, document)
                answer += [item for item in items if item not in answer]
                citations += cited
        result["sweep"] = {"kind_search": decision["kind_search"], "condition": condition, "documents": documents}
        if answer:
            result.update(answer=answer, citations=citations, swept=True, iterations=int(draft.get("iterations", 1)) + 1)
        return result

    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, drafts))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.drafts.with_name(f"{args.drafts.stem}{suffix}_sweep.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(f"Swept {sum(o['swept'] for o in outputs)} of {len(outputs)} "
          f"(asked {sum('sweep' in o for o in outputs)}); seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
