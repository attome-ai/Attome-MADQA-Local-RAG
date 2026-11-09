"""Hybrid read: the top pages as images, many more pages of the same documents as OCR text, in one request.

    python -m lab.hybrid experiments/thin_ocr_visual.json --model NAME --split dev --ids set.json

Images cost about 3,200 tokens a page, so the reader sees 9 pages. OCR text costs a few hundred tokens a page,
so the same context holds dozens. For questions whose answer is spread over many pages, the reader sees the
whole spread at once and can judge whether its list is complete. Rows match lab.answer's output.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import PROMPT, parse_reply, saved_rankings
from .read import RULES, call_server, render_jpeg
from .run import read_jsonl, select_questions, sha256

NOTE = """Pages 1 to {images} are given as images. After them comes the extracted text of more pages from the same
documents ("[file page N]"), which may contain OCR errors; use it to find items the images do not show, and check
any item against its page image when the page is among the images."""


def load_text(pages_path: Path) -> dict[tuple[str, int], str]:
    return {(r["document"], int(r["page"])): r.get("text", "") for r in read_jsonl(pages_path)}


def text_pages(ranked: list[tuple[str, int]], shown: list, text: dict, budget: int) -> list[tuple[str, int]]:
    """Pages of the top documents, ranked pages first, then the rest in reading order, within a character budget."""
    docs = list(dict.fromkeys(d for d, _ in ranked[:3]))
    order = [p for p in ranked if p[0] in docs and p not in shown]
    rest = sorted((p for p in text if p[0] in docs and p not in shown and p not in order), key=lambda p: (docs.index(p[0]), p[1]))
    picked, used = [], 0
    for page in order + rest:
        size = len(text.get(page, ""))
        if used + size > budget:
            continue
        picked.append(page)
        used += size
    return sorted(picked, key=lambda p: (docs.index(p[0]), p[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--budget", type=int, default=50000, help="characters of page text")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    text = load_text(pages_path)
    questions = select_questions(read_jsonl(args.data / "questions" / f"{args.split}.jsonl"), args.ids)

    def one(q: dict) -> dict:
        shown = ranked[q["id"]][: args.images]
        more = text_pages(ranked[q["id"]], shown, text, args.budget)
        content: list[dict] = []
        for n, (doc, page) in enumerate(shown, 1):
            content.append({"type": "text", "text": f"Page {n} (file {doc}, its page {page}):"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{render_jpeg(args.data / 'pdfs' / doc, page, args.max_side)}"}})
        extracted = "\n\n".join(f"[{d} page {p}]\n{text[(d, p)]}" for d, p in more)
        prompt = PROMPT.format(count=len(shown), question=q["question"])
        content.append({"type": "text", "text": f"{RULES['v3']}\n\n{NOTE.format(images=len(shown))}\n\n{extracted}\n\n{prompt}"})
        try:
            reply = call_server(args.base_url, args.model, content)
            answer, citations = parse_reply(reply["choices"][0]["message"]["content"] or "", shown)
        except Exception as error:  # one bad request must not lose the run
            print("FAILED", q["id"], error, flush=True)
            answer, citations = [], []
        return {"question": q["question"], "answer": answer, "citations": citations, "iterations": 1, "id": q["id"], "text_pages": len(more)}

    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(one, questions))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.data / "results" / f"hybrid_{config['name']}_{args.split}{suffix}.jsonl"
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print("saved", path, "avg text pages", sum(o["text_pages"] for o in out) / max(1, len(out)))


if __name__ == "__main__":
    main()
