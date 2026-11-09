"""Read in stages: a fast read for every question, and a full read only when it looks unsure.

    python -m lab.cascade experiments/thin_ocr_fast.json data/results/answers_full.jsonl --model NAME

To save time, this reuses saved full-read answers and only runs the fast read. The combined file
scores exactly what a live system would produce, and the report estimates its average time.
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

CONFIDENCE_RULE = """Also report whether you are sure. Say "confident": false if the pages don't clearly
hold the answer, if the question needs evidence you can't see, or if you had to guess, calculate
across many items, or read small or unclear text. Add it to your JSON reply, for example:
{"answer": ["..."], "pages": [1], "confident": true}"""


def needs_full_read(answer: list[str], confident: object) -> bool:
    """Anything except a clear yes from the model goes to the full read."""
    return confident is not True or looks_unanswered(answer)


def read_fast(
    question: dict,
    pages: list,
    call: Callable[[list[dict]], dict],
    data: Path,
    rules: str,
    max_side: int,
) -> dict:
    content = build_content(question["question"], pages, data, f"{rules}\n{CONFIDENCE_RULE}", max_side, True)
    with timed("fast_read"):
        try:
            reply = call(content)
        except (urllib.error.URLError, OSError) as error:
            print(f"REQUEST FAILED {question['id']}: {error}", flush=True)
            return {"answer": [], "citations": [], "confident": False}
    text = reply["choices"][0]["message"]["content"] or ""
    data_json, _ = final_json(text)
    answer, citations = parse_reply(text, pages)
    confident = data_json.get("confident") if data_json else None
    return {"answer": answer, "citations": citations, "confident": confident}


def combine(fast: dict, full: dict) -> dict:
    """Use the fast read when it's sure, otherwise the full read, and count iterations honestly."""
    escalated = needs_full_read(fast["answer"], fast["confident"])
    if escalated:
        return {**full, "iterations": 2, "escalated": True, "fast_answer": fast["answer"]}
    return {**full, "answer": fast["answer"], "citations": fast["citations"], "iterations": 1, "escalated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("full_answers", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=1568)
    parser.add_argument("--rules", default="v3", choices=sorted(RULES))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-seconds", type=float, default=40.0, help="measured time of one full read")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    full = read_jsonl(args.full_answers)
    reset()
    start = time.perf_counter()
    call = lambda content: call_server(args.base_url, args.model, content)

    def one(row: dict) -> dict:
        return combine(
            read_fast(row, ranked[row["id"]][: args.top_k], call, args.data, RULES[args.rules], args.max_side), row
        )

    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, full))
    seconds = time.perf_counter() - start
    path = args.full_answers.with_name(f"{args.full_answers.stem}_cascade.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    share = sum(o["escalated"] for o in outputs) / len(outputs)
    fast_mean = snapshot()["fast_read"]["mean_ms"] / 1000
    print(path)
    print(
        f"Sent to full read: {100 * share:.1f}%; fast read {fast_mean:.1f}s average; "
        f"estimated average {fast_mean + share * args.full_seconds:.1f}s vs {args.full_seconds:.1f}s; "
        f"fast pass took {seconds:.0f}s"
    )
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
