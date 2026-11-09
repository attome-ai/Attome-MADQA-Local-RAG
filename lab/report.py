"""Print every experiment, best first:  python -m lab.report --split dev --k 10"""

from __future__ import annotations

import argparse
from pathlib import Path

from .run import read_jsonl, sha256


def report(data: Path, split: str, k: int) -> str:
    ledger = data / "results" / "ledger.jsonl"
    entries = read_jsonl(ledger) if ledger.exists() else []
    entries = [e for e in entries if e["split"] == split and e["limit"] is None]
    # Older runs on a different version of a text source are not comparable, so hide them.
    latest = {}
    for entry in entries:
        path = data / "pages" / f"{entry['pages']}.jsonl"
        latest.setdefault(entry["pages"], sha256(path) if path.exists() else None)
    entries = [e for e in entries if e.get("pages_hash") == latest[e["pages"]]]
    # A forced rerun replaces the earlier record of the same run.
    entries = list({e["key"]: e for e in entries}.values())
    metric = f"page_recall@{k}"
    entries = [e for e in entries if metric in e["metrics"]["overall"]]
    entries.sort(key=lambda e: -e["metrics"]["overall"][metric])
    lines = [f"{metric:>16} {'all-ev':>7} {'single':>7} {'x-page':>7} {'x-doc':>7} {'sec/q':>8}  name"]
    for e in entries:
        m = e["metrics"]
        all_evidence = m["overall"][f"all_evidence@{k}"]
        by_hop = [m[hop].get(metric, float("nan")) for hop in ("single", "cross_page", "cross_doc")]
        lines.append(
            f"{m['overall'][metric]:16.2f} {all_evidence:7.2f} {by_hop[0]:7.2f} {by_hop[1]:7.2f} "
            f"{by_hop[2]:7.2f} {e['seconds_per_question']:8.4f}  {e['name']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(report(args.data, args.split, args.k))
