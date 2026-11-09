"""Thinking when unsure: re-read with thinking only where two independent answers disagree.

    python -m lab.escalate first.jsonl second.jsonl experiments/thin_ocr_llm_select.json --model NAME

Thinking was over 3x slower when used on every question, so this spends it only on likely mistakes.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .answer import build_content, parse_reply, saved_rankings
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server
from .run import read_jsonl, sha256


def normalized(answer: list) -> list[str]:
    """Compare answers ignoring case, spacing, punctuation and list order."""
    return sorted(re.sub(r"[^a-z0-9]", "", str(item).lower()) for item in answer)


def disagree(first: list, second: list) -> bool:
    return normalized(first) != normalized(second)


def escalate_questions(
    first: list[dict],
    second: dict[str, dict],
    ranked: dict[str, list],
    call: Callable[[list[dict]], dict],
    data: Path,
    top_k: int,
    rules: str,
    max_side: int | None,
    workers: int = 4,
) -> list[dict]:
    def one(row: dict) -> dict:
        other = second.get(row["id"])
        pages = ranked[row["id"]][:top_k]
        if other is None or not pages or not disagree(row["answer"], other["answer"]):
            return {**row, "escalated": False}
        content = build_content(row["question"], pages, data, rules, max_side, True)
        with timed("escalate_total"):
            try:
                reply = call(content)
            except (urllib.error.URLError, OSError) as error:
                print(f"REQUEST FAILED {row['id']}: {error}", flush=True)
                return {**row, "escalated": False}
        answer, citations = parse_reply(reply["choices"][0]["message"]["content"] or "", pages)
        # Reasoning cut off at the token limit has no final answer, so the first answer stands.
        if not answer or len(" ".join(answer)) > 1000:
            return {**row, "escalated": True, "escalation_unusable": True}
        result = {**row, "escalated": True, "before_escalation": row["answer"], "answer": answer}
        if citations:
            result["citations"] = citations
        return result

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, first))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--rules", default="v3", choices=sorted(RULES))
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    second = {row["id"]: row for row in read_jsonl(args.second)}
    reset()
    start = time.perf_counter()
    outputs = escalate_questions(
        read_jsonl(args.first),
        second,
        ranked,
        lambda content: call_server(args.base_url, args.model, content, thinking=True),
        args.data,
        args.top_k,
        RULES[args.rules],
        args.max_side,
        args.workers,
    )
    seconds = time.perf_counter() - start
    path = args.first.with_name(f"{args.first.stem}_escalated.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    escalated = sum(o["escalated"] for o in outputs)
    unusable = sum(bool(o.get("escalation_unusable")) for o in outputs)
    print(path)
    print(f"Escalated {escalated} of {len(outputs)}, unusable {unusable}; seconds {seconds:.0f}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
