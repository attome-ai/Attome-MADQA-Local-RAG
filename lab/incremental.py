"""Rerun only what a retrieval change touches.

    python -m lab.incremental changed old_config.json new_config.json --split dev --out experiments/changed.json
    python -m lab.incremental merge old_answers.jsonl rerun_answers.jsonl out.jsonl

`changed` lists the questions whose first k ranked pages differ, so readers need to see them again.
`merge` keeps the old row for every other question. Readers aren't perfectly repeatable, so this is an
estimate of a full rerun, and the copied rows are the same ones the baseline was judged on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import saved_rankings
from .run import read_jsonl, sha256


def changed_ids(old: dict[str, list], new: dict[str, list], k: int) -> list[str]:
    return [i for i in old if i in new and [tuple(p) for p in old[i][:k]] != [tuple(p) for p in new[i][:k]]]


def merge_rows(old: list[dict], rerun: list[dict]) -> list[dict]:
    fresh = {row["id"]: row for row in rerun}
    return [fresh.get(row["id"], row) for row in old]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    changed = sub.add_parser("changed")
    changed.add_argument("old", type=Path)
    changed.add_argument("new", type=Path)
    changed.add_argument("--split", default="dev")
    changed.add_argument("--k", type=int, default=9)
    changed.add_argument("--out", type=Path, required=True)
    changed.add_argument("--data", type=Path, default=Path("data"))
    merge = sub.add_parser("merge")
    merge.add_argument("old", type=Path)
    merge.add_argument("rerun", type=Path)
    merge.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "changed":
        rankings = []
        for path in (args.old, args.new):
            config = json.loads(path.read_text(encoding="utf-8"))
            pages = args.data / "pages" / f"{config['pages']}.jsonl"
            rankings.append(saved_rankings(args.data, config["name"], args.split, sha256(pages)))
        ids = changed_ids(*rankings, args.k)
        purpose = f"Questions whose top {args.k} pages differ between {args.old.stem} and {args.new.stem} ({args.split})"
        args.out.write_text(json.dumps({"purpose": purpose, "ids": {"changed": ids}}, indent=1), encoding="utf-8")
        print(f"Changed {len(ids)} of {len(rankings[0])}")
    else:
        rows = merge_rows(read_jsonl(args.old), read_jsonl(args.rerun))
        with args.out.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"Merged {len(rows)} rows into {args.out}")


if __name__ == "__main__":
    main()
