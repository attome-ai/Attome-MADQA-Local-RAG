"""Verify and fix: re-read the same pages, quote the evidence, and correct the draft if it fails.

    python -m lab.verify data/results/answers_x.jsonl experiments/thin_ocr_llm_select.json \
        --model NAME --top-k 9 --max-side 2048

The draft stands whenever the check fails or can't be read, so this step can't lose an answer.
Saved as a new file ending in _verified.jsonl, ready for lab.judge.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .agent import looks_unanswered
from .answer import build_content, parse_reply, saved_rankings
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server, final_json
from .run import read_jsonl, sha256

VERIFY_PROMPT = """A draft answer to this question is below. Check it against the pages above.
1. Find the exact row, line or passage that answers the question, and quote it.
2. Check the quote really matches what was asked: the right person, year, row and column.
3. If the question needs a count, total, comparison or date calculation, redo it step by step.
4. If the draft is right, keep it unchanged. If it is wrong, replace it. Keep the same short format.
Reply with JSON only:
{{"quote": "...", "draft_correct": true, "answer": ["..."], "pages": [1]}}

Draft answer: {draft}"""


def merge_verdict(draft: dict, text: str, pages: list) -> dict:
    """Apply the check to the draft row, keeping the draft unless there's a usable correction."""
    data, _ = final_json(text)
    result = {**draft, "draft_answer": draft["answer"], "changed": False}
    if data is None:
        return result
    answer, citations = parse_reply(json.dumps(data), pages)
    answer = [item for item in answer if item.strip()]
    if not answer:
        return result
    result["quote"] = str(data.get("quote", ""))
    if data.get("draft_correct") is False and answer != draft["answer"] and not looks_unanswered(answer):
        result["answer"], result["changed"] = answer, True
        # Citations change only with the answer, because a kept draft keeps its own evidence.
        if citations:
            result["citations"] = citations
    return result


def verify_questions(
    drafts: list[dict],
    ranked: dict[str, list],
    call: Callable[[list[dict]], dict],
    data: Path,
    top_k: int,
    rules: str,
    max_side: int | None,
    workers: int = 6,
) -> list[dict]:
    def one(draft: dict) -> dict:
        pages = ranked[draft["id"]][:top_k]
        if not pages or not draft["answer"]:
            return {**draft, "draft_answer": draft["answer"], "changed": False}
        content = build_content(draft["question"], pages, data, rules, max_side, True)
        prompt = VERIFY_PROMPT.format(draft=json.dumps(draft["answer"], ensure_ascii=False))
        content[-1] = {"type": "text", "text": f"{content[-1]['text']}\n\n{prompt}"}
        with timed("verify_total"):
            try:
                reply = call(content)
            except (urllib.error.URLError, OSError) as error:
                print(f"REQUEST FAILED {draft['id']}: {error}", flush=True)
                return {**draft, "draft_answer": draft["answer"], "changed": False}
        return merge_verdict(draft, reply["choices"][0]["message"]["content"] or "", pages)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, drafts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--rules", default="v3", choices=sorted(RULES))
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    drafts = read_jsonl(args.answers)
    reset()
    start = time.perf_counter()
    outputs = verify_questions(
        drafts,
        ranked,
        lambda content: call_server(args.base_url, args.model, content),
        args.data,
        args.top_k,
        RULES[args.rules],
        args.max_side,
        args.workers,
    )
    seconds = time.perf_counter() - start
    path = args.answers.with_name(f"{args.answers.stem}_verified.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    changed = sum(o["changed"] for o in outputs)
    print(path)
    print(f"Changed {changed} of {len(outputs)}; questions per second {len(outputs) / seconds:.3f}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
