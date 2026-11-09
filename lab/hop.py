"""Second hop inside the document (T32): find the fact the first read was missing, then read again.

    python -m lab.hop drafts_verified.jsonl experiments/thin_ocr_llm_select.json --model NAME --split dev

Two-step questions inside one document ("the offices of the company where X works") often show the first
page but not the second: it matches the bridge fact, not the question, so it ranks low. After the first read,
the model says whether a fact is still missing and what to search for. BM25 then searches only the documents
already shown, and the reader re-reads with up to 3 new pages swapped in for the last shown ones.
Questions with nothing missing keep the draft. Output is a candidate for lab.adjudicate, ending in _hop.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import build_content, parse_reply, saved_rankings
from .agent import looks_unanswered
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server
from .run import read_jsonl, sha256
from .stages import PageIndex

HOP_PROMPT = """Question: {question}
Answer found so far: {answer}
Evidence quoted: {quote}

Does answering the question fully need another fact that this evidence doesn't give, such as a detail about a
person, company, date or item that the evidence only names? If yes, write a short keyword search for that
missing fact, using the names from the evidence. If the evidence already answers the question, say no.
Reply with JSON only: {{"missing": true, "search": "..."}}"""


def parse_hop(text: str) -> str | None:
    body = text.rsplit("</think>", 1)[-1]
    start, end = body.find("{"), body.rfind("}")
    try:
        data = json.loads(body[start : end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("missing") is not True:
        return None
    search = str(data.get("search", "")).strip()
    return search or None


def hop_pages(shown: list, found: list, keep: int, swap: int) -> list:
    new = [page for page in found if page not in shown][:swap]
    return shown[:keep] + new if new else shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--swap", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    index = PageIndex(args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite")
    index.pages_path = pages_path
    drafts = read_jsonl(args.drafts)
    call = lambda content: call_server(args.base_url, args.model, content)

    def one(draft: dict) -> dict:
        result = {**draft, "hop_search": None, "hopped": False}
        shown = [tuple(p) for p in ranked[draft["id"]][: args.top_k]]
        if not shown:
            return result
        prompt = HOP_PROMPT.format(
            question=draft["question"],
            answer=json.dumps(draft["answer"], ensure_ascii=False),
            quote=json.dumps(draft.get("quote", ""), ensure_ascii=False)[:600],
        )
        with timed("hop_decide"):
            try:
                reply = call([{"type": "text", "text": prompt}])
            except (urllib.error.URLError, OSError):
                return result
        search = parse_hop(reply["choices"][0]["message"]["content"] or "")
        if search is None:
            return result
        documents = {document for document, _ in shown[:3]} | {c["file"] for c in draft.get("citations", [])}
        found = [page for page in index.bm25(search, 3000) if page[0] in documents]
        pages = hop_pages(shown, found, args.top_k - args.swap, args.swap)
        result["hop_search"] = search
        if pages == shown:
            return result
        content = build_content(draft["question"], pages, args.data, RULES["v3"], args.max_side, True)
        with timed("hop_read"):
            try:
                reply = call(content)
            except (urllib.error.URLError, OSError) as error:
                print(f"REQUEST FAILED {draft['id']}: {error}", flush=True)
                return result
        answer, citations = parse_reply(reply["choices"][0]["message"]["content"] or "", pages)
        if answer and not looks_unanswered(answer):
            result.update(answer=answer, citations=citations, hopped=True, iterations=int(draft.get("iterations", 1)) + 1)
        return result

    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, drafts))
    path = args.drafts.with_name(f"{args.drafts.stem}_hop.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    asked = sum(o["hop_search"] is not None for o in outputs)
    print(f"Hop wanted {asked}, re-read {sum(o['hopped'] for o in outputs)} of {len(outputs)}; "
          f"seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
