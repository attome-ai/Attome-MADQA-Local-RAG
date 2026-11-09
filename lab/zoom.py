"""Zoom (T7): re-read the cited pages at higher resolution, as an extra candidate for lab.adjudicate.

    python -m lab.zoom drafts.jsonl experiments/thin_ocr_llm_select.json --model NAME --split dev

Misreads of small print ("66" read as "64") happen on the page the reader thinks holds the answer. This reads
that page (up to 3 cited pages, topped up from the ranking) at a larger size. Three large pages cost fewer
tokens than the usual nine. Saved as a new file ending in _zoom.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import build_content, parse_reply, saved_rankings
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server
from .run import read_jsonl, select_questions, sha256


def zoom_pages(draft: dict, ranked: list, count: int) -> list:
    pages = [(c["file"], int(c["page"])) for c in draft.get("citations", [])]
    for page in ranked:
        if len(pages) >= count:
            break
        if tuple(page) not in pages:
            pages.append(tuple(page))
    return list(dict.fromkeys(pages))[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drafts", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=2800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    drafts = select_questions(read_jsonl(args.drafts), args.ids)
    call = lambda content: call_server(args.base_url, args.model, content)

    def one(draft: dict) -> dict:
        pages = zoom_pages(draft, ranked[draft["id"]], args.pages)
        result = {**draft, "answer": [], "citations": [], "zoom_pages": pages}
        if not pages:
            return result
        content = build_content(draft["question"], pages, args.data, RULES["v3"], args.max_side, True)
        with timed("zoom_read"):
            try:
                reply = call(content)
            except (urllib.error.URLError, OSError) as error:
                print(f"REQUEST FAILED {draft['id']}: {error}", flush=True)
                return result
        result["answer"], result["citations"] = parse_reply(reply["choices"][0]["message"]["content"] or "", pages)
        return result

    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, drafts))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.drafts.with_name(f"{args.drafts.stem}{suffix}_zoom.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(f"Zoomed {len(outputs)}; seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
