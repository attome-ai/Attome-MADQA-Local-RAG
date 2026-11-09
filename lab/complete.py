"""Complete lists (T14): for list questions, re-read the cited pages plus nearby pages of the same documents.

    python -m lab.complete data/results/answers_x.jsonl experiments/thin_ocr_llm_select.json --model NAME

Lists often continue on the next page or sit in several places, and the reader tends to stop early.
Only list questions are re-read. The draft stands whenever the reply can't be used.
Saved as a new file ending in _completed.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .agent import looks_unanswered
from .answer import build_content, parse_reply, saved_rankings
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server, final_json
from .run import read_jsonl, select_questions, sha256

PLURAL = re.compile(
    r"^\s*(what|which|who)\s+(are|were)\b|\b(names|list|titles|categories|all|members|types|kinds)\b",
    re.IGNORECASE,
)

PROMPT = """A draft answer to this list question is below. Lists in these documents often continue on the next
page, or appear in several places. Go through every page shown and collect every item that answers the
question, with its page. Then remove items that don't answer exactly what was asked. If the draft is
already complete and correct, keep it unchanged. Keep the same short format for each item.
Reply with JSON only:
{{"found": [{{"item": "...", "page": 1}}], "answer": ["..."], "pages": [1]}}

Draft answer: {draft}"""


def is_list_question(question: str, answer: list) -> bool:
    return len(answer) >= 2 or bool(PLURAL.search(question))


@lru_cache(maxsize=4096)
def page_count(path: Path) -> int:
    import pypdfium2

    pdf = pypdfium2.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def nearby_pages(
    cited: list[tuple[str, int]],
    ranked: list[tuple[str, int]],
    counts: Callable[[str], int],
    radius: int,
    limit: int,
) -> list[tuple[str, int]]:
    """Cited pages first, then pages within the radius in the same documents, then the ranking."""
    pages: list[tuple[str, int]] = []
    add = lambda p: pages.append(p) if p not in pages else None
    for page in cited:
        add(page)
    for offset in range(1, radius + 1):
        for document, number in cited:
            for candidate in (number + offset, number - offset):
                if 1 <= candidate <= counts(document):
                    add((document, candidate))
    for page in ranked:
        add(page)
    return pages[:limit]


def complete_one(
    draft: dict,
    ranked: list,
    call: Callable[[list[dict]], dict],
    data: Path,
    rules: str,
    max_side: int | None,
    radius: int,
    limit: int,
) -> dict:
    result = {**draft, "list_answer": draft["answer"], "completed": False}
    if looks_unanswered(draft["answer"]) or not is_list_question(draft["question"], draft["answer"]):
        return result
    cited = [(c["file"], int(c["page"])) for c in draft.get("citations", [])]
    counts = lambda document: page_count(data / "pdfs" / document)
    pages = nearby_pages(cited, [tuple(p) for p in ranked], counts, radius, limit)
    if not pages:
        return result
    content = build_content(draft["question"], pages, data, rules, max_side, True)
    prompt = PROMPT.format(draft=json.dumps(draft["answer"], ensure_ascii=False))
    content[-1] = {"type": "text", "text": f"{content[-1]['text']}\n\n{prompt}"}
    with timed("complete"):
        try:
            reply = call(content)
        except (urllib.error.URLError, OSError) as error:
            print(f"REQUEST FAILED {draft['id']}: {error}", flush=True)
            return result
    text = reply["choices"][0]["message"]["content"] or ""
    parsed, _ = final_json(text)
    if parsed is None:
        return result
    answer, citations = parse_reply(json.dumps(parsed), pages)
    answer = [item for item in answer if item.strip()]
    if not answer or looks_unanswered(answer):
        return result
    result["iterations"] = int(draft.get("iterations", 1)) + 1
    if answer != draft["answer"]:
        result["answer"], result["completed"] = answer, True
        if citations:
            result["citations"] = citations
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path, help="only re-read the questions in this id-set file")
    parser.add_argument("--rules", default="v3", choices=sorted(RULES))
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--limit", type=int, default=9, help="9 pages at 2,048 pixels fit the 32k context; 12 do not")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    drafts = select_questions(read_jsonl(args.answers), args.ids)
    reset()
    start = time.perf_counter()
    call = lambda content: call_server(args.base_url, args.model, content)

    def one(draft: dict) -> dict:
        return complete_one(
            draft, ranked[draft["id"]], call, args.data, RULES[args.rules], args.max_side, args.radius, args.limit
        )

    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, drafts))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.answers.with_name(f"{args.answers.stem}{suffix}_completed.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    listed = sum(o["list_answer"] is not None and is_list_question(o["question"], o["list_answer"]) for o in outputs)
    print(f"List questions {listed}, changed {sum(o['completed'] for o in outputs)} of {len(outputs)}; "
          f"seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
