"""Page notes: read each lower-ranked page alone, keep what bears on the question as text, answer once.

    python -m lab.notes answers.jsonl experiments/thin_ocr_visual.json --model NAME --split dev [--ids set.json]

The reader sees 9 page images, so pages ranked 10-40 never reach it (86% of multi-page questions have every gold
page within the top 40, 72% within 9). Here the top --images pages stay images; each of the next --notes pages is
read alone and turned into notes (facts quoted as written, no answer). One final read sees the images plus all
notes and decides. Output ends in _notes.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import build_content, saved_rankings
from .read import RULES_V3, call_server, final_json
from .run import read_jsonl, select_questions, sha256

NOTE = """Question (do not answer it): {question}

Write notes from this page for someone who will answer the question later and cannot see the page. Quote every
fact on the page that could help, exactly as written: names, numbers, dates, table rows with their column
headings, list entries. Include the page's title or the document's subject in one line. If nothing on the page
bears on the question, reply with the single word NONE. Plain text, no answer."""

BLOCK = """Notes taken from other pages of the documents (each page was read alone; they may be incomplete):
{notes}

Use the page images and these notes together."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument("--notes", type=int, default=25, help="pages after the images turned into notes")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    rows = select_questions(read_jsonl(args.answers), args.ids)

    def call(content: list[dict]) -> str:
        try:
            reply = call_server(args.base_url, args.model, content)
        except (urllib.error.URLError, OSError):
            time.sleep(2)
            reply = call_server(args.base_url, args.model, content)
        return (reply["choices"][0]["message"]["content"] or "").rsplit("</think>", 1)[-1].strip()

    def note(question: str, page: tuple[str, int]) -> str:
        content = build_content(question, [page], args.data, "", args.max_side, True)
        content[-1] = {"type": "text", "text": NOTE.format(question=question)}
        try:
            return call(content)
        except (urllib.error.URLError, OSError):
            return "NONE"

    def one(row: dict) -> dict:
        result = {**row, "notes_answer": row["answer"], "notes_used": False}
        pages = ranked[row["id"]]
        shown, rest = pages[: args.images], pages[args.images : args.images + args.notes]
        notes = []
        for page in rest:
            text = note(row["question"], page)
            if text and text.upper().strip(". ") != "NONE":
                notes.append(f"[{page[0]} page {page[1]}] {text}")
        result["notes_count"] = len(notes)
        if not notes:
            return result
        rules = RULES_V3 + "\n\n" + BLOCK.format(notes="\n".join(notes))
        content = build_content(row["question"], shown, args.data, rules, args.max_side, True)
        try:
            data = final_json(call_server(args.base_url, args.model, content)["choices"][0]["message"]["content"] or "")[0]
        except (urllib.error.URLError, OSError):
            return result
        answer = [str(a) for a in (data or {}).get("answer", []) if str(a).strip()] if isinstance(data, dict) else []
        if answer and answer != row["answer"]:
            result.update(answer=answer, notes_used=True)
        return result

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, rows))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.answers.with_name(f"{args.answers.stem}{suffix}_notes.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(f"questions {len(outputs)}; with notes {sum(o.get('notes_count', 0) > 0 for o in outputs)}; "
          f"answer changed {sum(o['notes_used'] for o in outputs)}; seconds {time.perf_counter() - start:.0f}; saved {path}")


if __name__ == "__main__":
    main()
