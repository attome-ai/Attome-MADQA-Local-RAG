"""Turn saved answers into a leaderboard file, and refuse anything that wouldn't be valid.

    python -m lab.submit data/results/answers_x_test.jsonl --out submission.jsonl

Writing the file sends nothing anywhere. Submitting to the leaderboard is a separate, manual step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import read_jsonl


def to_submission_row(output: dict) -> dict:
    return {
        "question": output["question"],
        "answer": [str(item) for item in output["answer"]],
        "citations": [{"file": c["file"], "page": int(c["page"])} for c in output["citations"]],
        "iterations": int(output.get("iterations", 1)),
        "id": output["id"],
    }


def validate(rows: list[dict], questions: list[dict], known_pages: set[tuple[str, int]]) -> list[str]:
    """Return every problem found, so one run reports them all at once."""
    problems = []
    expected = {q["id"]: q for q in questions}
    seen = [row["id"] for row in rows]
    missing = set(expected) - set(seen)
    if missing:
        problems.append(f"{len(missing)} questions unanswered, e.g. {sorted(missing)[:3]}")
    extra = set(seen) - set(expected)
    if extra:
        problems.append(f"{len(extra)} unknown ids, e.g. {sorted(extra)[:3]}")
    if len(seen) != len(set(seen)):
        problems.append("duplicate ids")
    for row in rows:
        question = expected.get(row["id"])
        if question and question["question"] != row["question"]:
            problems.append(f"{row['id']}: question text differs")
        if not row["answer"]:
            problems.append(f"{row['id']}: empty answer")
        if row["iterations"] < 1:
            problems.append(f"{row['id']}: iterations below 1")
        for citation in row["citations"]:
            if (citation["file"], citation["page"]) not in known_pages:
                problems.append(f"{row['id']}: cites unknown page {citation}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    rows = [to_submission_row(o) for o in read_jsonl(args.answers)]
    questions = read_jsonl(args.data / "questions" / f"{args.split}.jsonl")
    pages = read_jsonl(args.data / "pages" / "pdf_text.jsonl")
    problems = validate(rows, questions, {(p["document"], p["page"]) for p in pages})
    if problems:
        print(f"{len(problems)} problems, no file written:")
        print("\n".join(problems[:30]))
        raise SystemExit(1)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    print(f"Wrote {len(rows)} valid rows to {args.out}")


if __name__ == "__main__":
    main()
