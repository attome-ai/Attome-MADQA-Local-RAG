"""SAG-inspired page context: give every page its document's opening text for search.

    python -m lab.contextualize --source hybrid_rapidocr --chars 400

A page often omits the name of its document or subject, such as an exhibit that never names
its agreement, so keyword search cannot reach it. Only search text changes. The reader still
sees the original page image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_context(pages: list[dict], chars: int) -> list[dict]:
    openings: dict[str, str] = {}
    for page in sorted(pages, key=lambda p: (p["document"], p["page"])):
        if page["document"] not in openings or len(openings[page["document"]]) < chars:
            joined = f"{openings.get(page['document'], '')} {page['text']}".strip()
            openings[page["document"]] = " ".join(joined.split())[:chars]
    return [
        {**page, "text": f"{openings[page['document']]}\n\n{page['text']}"} for page in pages
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="hybrid_rapidocr")
    parser.add_argument("--chars", type=int, default=400)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    folder = args.data / "pages"
    with (folder / f"{args.source}.jsonl").open(encoding="utf-8") as handle:
        pages = [json.loads(line) for line in handle]
    result = add_context(pages, args.chars)
    target = folder / f"{args.source}_ctx{args.chars}.jsonl"
    with target.open("w", encoding="utf-8") as out:
        out.writelines(json.dumps(page, ensure_ascii=False) + "\n" for page in result)
    print(f"Wrote {len(result)} pages to {target}")


if __name__ == "__main__":
    main()
