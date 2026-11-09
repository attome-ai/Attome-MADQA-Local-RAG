"""Experiment 3: OCR only the pages with almost no embedded text, on the GPU.

    docker build -t madqa-ocr -f docker/ocr.Dockerfile .
    docker run --rm --gpus all -v "<project>/lab:/app/lab" -v "<project>/data:/app/data" \
        -v madqa-ocr-models:/models madqa-ocr --limit 16

Fails rather than falling back to the CPU if the GPU is unavailable. Finished pages are saved
as they complete, so an interrupted run resumes. --limit runs a timed sample only, and the
merged text source data/pages/hybrid_rapidocr.jsonl is written only after a complete run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .profiling import format_table, profiled, snapshot

MIN_TEXT = 20
MAX_SIDE = 2000


def merge(pages: list[dict], ocr: dict[tuple[str, int], str], min_text: int = MIN_TEXT) -> list[dict]:
    """Add OCR text to thin pages, keeping any embedded text such as a stamp over a scan."""
    merged = []
    for page in pages:
        embedded = page["text"].strip()
        found = ocr.get((page["document"], page["page"]), "")
        if len(embedded) < min_text and found:
            text = f"{embedded}\n{found}".strip() if len(embedded) >= MIN_TEXT else found
            merged.append({**page, "text": text})
        else:
            merged.append(page)
    return merged


@profiled
def render_page(path: Path, page: int):
    import numpy as np
    import pypdfium2

    pdf = pypdfium2.PdfDocument(path)
    try:
        pdf_page = pdf[page - 1]
        scale = min(2.0, MAX_SIDE / max(pdf_page.get_size()))
        return np.array(pdf_page.render(scale=scale).to_pil().convert("RGB"))
    finally:
        pdf.close()


def load_engine():
    import onnxruntime

    onnxruntime.preload_dlls()
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("CUDA is unavailable, refusing to run OCR on the CPU")
    from rapidocr import RapidOCR

    return RapidOCR(params={"EngineConfig.onnxruntime.use_cuda": True})


@profiled
def recognize(engine, image) -> str:
    return "\n".join(engine(image).txts or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-text", type=int, default=MIN_TEXT, help="OCR pages with less embedded text")
    parser.add_argument("--output", default="hybrid_rapidocr")
    args = parser.parse_args()

    with (args.data / "pages" / "pdf_text.jsonl").open(encoding="utf-8") as handle:
        pages = [json.loads(line) for line in handle]
    cache_path = args.data / "pages" / "rapidocr_cache.jsonl"
    done: dict[tuple[str, int], str] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                done[(row["document"], row["page"])] = row["text"]

    jobs = [
        (p["document"], p["page"])
        for p in pages
        if len(p["text"].strip()) < args.min_text and (p["document"], p["page"]) not in done
    ]
    remaining = len(jobs)
    jobs = jobs[: args.limit]
    print(f"Pages to OCR: {remaining}; this run: {len(jobs)}; already done: {len(done)}", flush=True)

    engine = load_engine()
    start = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as cache:
        for count, (document, page) in enumerate(jobs, 1):
            text = recognize(engine, render_page(args.data / "pdfs" / document, page))
            row = {"document": document, "page": page, "text": text}
            cache.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache.flush()
            done[(document, page)] = text
            if count % 100 == 0:
                print(f"{count}/{len(jobs)}", flush=True)

    if jobs:
        print(f"Seconds per page: {(time.perf_counter() - start) / len(jobs):.3f}")
    print(format_table(snapshot()))
    if len(jobs) == remaining:
        merged = merge(pages, done, args.min_text)
        with (args.data / "pages" / f"{args.output}.jsonl").open("w", encoding="utf-8") as out:
            out.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in merged)
        empty = sum(len(p["text"].strip()) < MIN_TEXT for p in merged)
        print(f"Pages: {len(merged)}; still near-empty after OCR: {empty}")


if __name__ == "__main__":
    main()
