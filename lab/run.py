"""Run one experiment config and record it in the ledger.

    python -m lab.run experiments/bm25_baseline.json --split dev

Identical runs (same config, questions and pages) are skipped, not repeated.
The test split is refused: its answers are hidden and it must not guide tuning.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import evaluate
from .profiling import format_table, reset, snapshot, timed
from .stages import PageIndex, run_pipeline


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_questions(questions: list[dict], ids_path: Path | None) -> list[dict]:
    """Keep only the questions named in an id-set file, failing on any unknown id."""
    if ids_path is None:
        return questions
    groups = json.loads(ids_path.read_text(encoding="utf-8"))["ids"]
    wanted = {question_id for group in groups.values() for question_id in group}
    missing = wanted - {q["id"] for q in questions}
    if missing:
        raise ValueError(f"Unknown question ids: {sorted(missing)}")
    return [q for q in questions if q["id"] in wanted]


def run_experiment(
    config_path: Path, split: str, data: Path, limit: int | None = None, force: bool = False
) -> dict:
    if split not in ("train", "dev"):
        raise ValueError("Only train and dev can be used for experiments")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    questions_path = data / "questions" / f"{split}.jsonl"
    pages_path = data / "pages" / f"{config['pages']}.jsonl"
    pages_hash = sha256(pages_path)
    key_source = {
        "pipeline": config["pipeline"],
        "eval_k": config["eval_k"],
        "split": split,
        "limit": limit,
        "pages": pages_hash,
        "questions": sha256(questions_path),
    }
    key = hashlib.sha256(json.dumps(key_source, sort_keys=True).encode()).hexdigest()[:16]

    ledger = data / "results" / "ledger.jsonl"
    if ledger.exists() and not force:
        for entry in read_jsonl(ledger):
            if entry["key"] == key:
                return {**entry, "cached": True}

    reset()
    index_path = data / "index" / f"{pages_hash[:16]}.sqlite"
    with timed("load_pages"):
        pages = read_jsonl(pages_path) if not index_path.exists() else None
    index = PageIndex(index_path, pages)
    index.pages_path = pages_path
    with timed("load_questions"):
        questions = read_jsonl(questions_path)[:limit]

    start = time.perf_counter()
    rows, predictions = [], []
    for question in questions:
        ranked = run_pipeline(question["question"], config["pipeline"], index)
        rows.append((question, ranked))
        predictions.append({"id": question["id"], "ranked": ranked[:100]})
    seconds = time.perf_counter() - start

    metrics = evaluate(rows, config["eval_k"])
    runs = data / "results" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(p) + "\n" for p in predictions)

    entry = {
        "key": key,
        "name": config["name"],
        "hypothesis": config.get("hypothesis", ""),
        "split": split,
        "limit": limit,
        "pages": config["pages"],
        "pages_hash": pages_hash,
        "pipeline": config["pipeline"],
        "seconds_per_question": round(seconds / max(len(questions), 1), 5),
        "metrics": metrics,
        "profile": snapshot(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return {**entry, "cached": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="rerun even if already recorded")
    parser.add_argument("--cprofile", type=Path, help="also save a full call profile to this file")
    args = parser.parse_args()
    profiler = cProfile.Profile() if args.cprofile else None
    if profiler:
        profiler.enable()
    entry = run_experiment(args.config, args.split, args.data, args.limit, args.force)
    if profiler:
        profiler.disable()
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.cprofile)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)
    print(f"{entry['name']} ({'cached' if entry['cached'] else 'new'})")
    print(json.dumps(entry["metrics"]["overall"], indent=2))
    print(format_table(entry.get("profile", {})))


if __name__ == "__main__":
    main()
