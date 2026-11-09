"""Download MADQA questions and PDFs, and extract each PDF's embedded text.

    python -m lab.prepare             # everything
    python -m lab.prepare --limit 10  # first 10 PDFs, as a smoke test

Requires accepting the dataset terms and `hf auth login`. Pages are stored 1-based.
Also reports how much text is missing and checks the gold page numbering.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.request import urlopen

REPO = "OxRML/MADQA"


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_questions(data: Path) -> None:
    from datasets import load_dataset

    for split in ("train", "dev"):
        write_jsonl(data / "questions" / f"{split}.jsonl", (dict(r) for r in load_dataset(REPO, split=split)))


def fetch_politely(
    names: list[str],
    urls: dict[str, str],
    out: Path,
    fetch: Callable[[str], bytes],
    pause: float = 3.0,
    max_blocked: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Download missing files one at a time, and stop if the server starts refusing us."""
    counts = {"downloaded": 0, "skipped": 0, "failed": 0, "blocked_stop": 0}
    blocked_in_a_row = 0
    for name in names:
        # Names come from the dataset, so keep only the final path component.
        target = out / Path(name).name
        if target.exists():
            counts["skipped"] += 1
            continue
        try:
            content = fetch(urls[name])
            temporary = target.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(target)
            counts["downloaded"] += 1
            blocked_in_a_row = 0
        except HTTPError as error:
            counts["failed"] += 1
            print(f"FAILED {name}: HTTP {error.code}")
            blocked_in_a_row = blocked_in_a_row + 1 if error.code in (403, 429) else 0
            if blocked_in_a_row >= max_blocked:
                counts["blocked_stop"] = 1
                print("Server is refusing requests. Stopping; rerun later to resume.")
                break
        except OSError as error:  # One dead link should not stop the rest.
            counts["failed"] += 1
            print(f"FAILED {name}: {error}")
        sleep(pause)
    return counts


def download_pdfs(data: Path, limit: int | None) -> None:
    from datasets import load_dataset

    links = load_dataset(REPO, "documents", split="links")
    urls = dict(sorted((r["document"], r["url"]) for r in links))
    out = data / "pdfs"
    out.mkdir(parents=True, exist_ok=True)

    def fetch(url: str) -> bytes:
        with urlopen(url, timeout=120) as response:
            return response.read()

    counts = fetch_politely(list(urls)[:limit], urls, out, fetch)
    print(f"PDFs on disk: {len(list(out.glob('*.pdf')))} of {len(urls)}; {counts}")


def extract_pages(data: Path) -> dict:
    import pypdfium2

    rows, unreadable = [], []
    for path in sorted((data / "pdfs").glob("*.pdf")):
        try:
            pdf = pypdfium2.PdfDocument(path)
        except Exception as error:
            unreadable.append(path.name)
            print(f"UNREADABLE {path.name}: {error}")
            continue
        for number in range(len(pdf)):
            text = pdf[number].get_textpage().get_text_range()
            rows.append({"document": path.name, "page": number + 1, "text": text})
        pdf.close()
    write_jsonl(data / "pages" / "pdf_text.jsonl", rows)
    empty = sum(len(row["text"].strip()) < 20 for row in rows)
    return {"pages": len(rows), "near_empty_pages": empty, "unreadable_pdfs": len(unreadable)}


def page_offset_report(questions: list[dict], pages: list[dict]) -> dict[int, float]:
    """How often an answer appears on the gold page, for each possible numbering offset.

    Offset 0 should win if gold numbering is 1-based. Offset 1 means gold is 0-based.
    """
    text = {(p["document"], p["page"]): re.sub(r"\s+", " ", p["text"]).lower() for p in pages}
    hits, total = defaultdict(int), 0
    for question in questions:
        variants = [str(a).lower() for group in question.get("answer_variants") or [] for a in group]
        variants = [a for a in variants if len(a) >= 4]
        for evidence in question.get("evidence") or []:
            if not variants or (evidence["document"], int(evidence["page"])) not in text:
                continue
            total += 1
            for offset in (-1, 0, 1):
                body = text.get((evidence["document"], int(evidence["page"]) + offset), "")
                hits[offset] += any(a in body for a in variants)
    return {offset: round(100 * hits[offset] / total, 2) if total else 0.0 for offset in (-1, 0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    download_questions(args.data)
    download_pdfs(args.data, args.limit)
    print(extract_pages(args.data))
    from .run import read_jsonl

    offsets = page_offset_report(
        read_jsonl(args.data / "questions" / "dev.jsonl"),
        read_jsonl(args.data / "pages" / "pdf_text.jsonl"),
    )
    print(f"Answer found on gold page, by numbering offset (%): {offsets}")


if __name__ == "__main__":
    main()
