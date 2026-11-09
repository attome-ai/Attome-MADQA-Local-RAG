"""Slot filling: when the question names things to look up and the answer lacks some of them, look them up.

    python -m lab.slots answers.jsonl experiments/thin_ocr_visual.json --model NAME --split dev [--ids set.json]

1. Text only: list the separate things the question names (years, people, items). Open-ended questions have none.
2. A slot counts as covered if its words appear in the current answer or quote. Nothing else is touched, so
   answers that already cover every slot stay exactly as they were.
3. Each uncovered slot gets its own search inside the documents the answer cites (plus the top-ranked one):
   the three best pages are read one at a time until one shows the value.
4. Text only: put the original answer and the found values together into the final answer.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import build_content, saved_rankings
from .docread import load_texts, same_subject
from .read import call_server, final_json
from .run import read_jsonl, select_questions, sha256
from .stages import PageIndex

SLOTS = """Question about a document: {question}

Does the answer have to contain one separate value for each of two or more things the question names (for example
"what was the revenue in 2019, 2020 and 2021" needs three values; "list the fee for A and for B" needs two)?
If so, give one short slot per value to report. Return an empty list when the question is a yes/no question, asks
to pick one of the named options ("which of A or B"), asks to compare or decide between them, asks for a single
fact, or asks for every item matching a condition.
Reply with JSON only: {{"slots": ["..."]}}"""

FILL = """Question: {question}

Look on this page for only this part of the answer: {slot}
If the page shows it, give the value exactly as written. If the page does not show it, say so.
Reply with JSON only: {{"found": true, "value": "..."}} or {{"found": false}}"""

COMPOSE = """Question: {question}

Current answer: {answer}
Values found by looking up the parts of the question separately: {found}

Give the final answer. Keep what the current answer already has; add the found values the question needs.
Answer in the same style as the current answer. Reply with JSON only: {{"answer": ["..."]}}"""


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def covered(slot: str, row: dict) -> bool:
    have = words(" ".join(map(str, row["answer"])) + " " + str(row.get("quote", "")))
    need = words(slot)
    return not need or need <= have


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--pages", type=int, default=3, help="pages read per slot")
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    index = PageIndex(args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite")
    index.pages_path = pages_path
    texts = load_texts(pages_path)
    rows = select_questions(read_jsonl(args.answers), args.ids)

    def call(content: list[dict]) -> dict:
        try:
            return call_server(args.base_url, args.model, content)
        except (urllib.error.URLError, OSError):
            time.sleep(2)
            return call_server(args.base_url, args.model, content)

    def reply(content: list[dict]) -> dict | None:
        try:
            body = (call(content)["choices"][0]["message"]["content"] or "").rsplit("</think>", 1)[-1]
        except (urllib.error.URLError, OSError):
            return None
        start, end = body.find("{"), body.rfind("}")
        try:
            data = json.loads(body[start : end + 1]) if start >= 0 else None
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def one(row: dict) -> dict:
        result = {**row, "slots_answer": row["answer"], "slots": []}
        data = reply([{"type": "text", "text": SLOTS.format(question=row["question"])}])
        slots = [str(s).strip() for s in (data or {}).get("slots", []) if str(s).strip()] if data else []
        result["slots"] = slots
        if len(slots) < 2:
            return result
        missing = [s for s in slots if not covered(s, row)]
        result["slots_missing"] = missing
        if not missing:
            return result
        cited = list(dict.fromkeys(c["file"] for c in row.get("citations", [])))
        documents = same_subject(row["question"], list(dict.fromkeys(cited + [ranked[row["id"]][0][0]])), texts)
        found = {}
        for slot in missing:
            candidates = [p for p in index.bm25(f"{slot} {row['question']}", 6000) if p[0] in documents][: args.pages]
            for page in candidates:
                content = build_content(row["question"], [page], args.data, "", args.max_side, True)
                content[-1] = {"type": "text", "text": FILL.format(question=row["question"], slot=slot)}
                got = reply(content)
                if got and got.get("found") is True and str(got.get("value", "")).strip():
                    found[slot] = f"{got['value']} (document {page[0]}, page {page[1]})"
                    break
        result["slots_found"] = found
        if not found:
            return result
        data = reply([{"type": "text", "text": COMPOSE.format(
            question=row["question"], answer=json.dumps(row["answer"], ensure_ascii=False),
            found=json.dumps(found, ensure_ascii=False))}])
        answer = [str(a) for a in (data or {}).get("answer", []) if str(a).strip()] if data else []
        if answer and answer != row["answer"]:
            result.update(answer=answer, slots_changed=True)
        return result

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(one, rows))
    suffix = f"_{args.ids.stem}" if args.ids else ""
    path = args.answers.with_name(f"{args.answers.stem}{suffix}_slots.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    multi = sum(len(o["slots"]) >= 2 for o in outputs)
    missing = sum(bool(o.get("slots_missing")) for o in outputs)
    changed = sum(bool(o.get("slots_changed")) for o in outputs)
    print(f"questions {len(outputs)}; with 2+ slots {multi}; with a missing slot {missing}; changed {changed}; "
          f"seconds {time.perf_counter() - start:.0f}; saved {path}")


if __name__ == "__main__":
    main()
