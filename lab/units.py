"""Units: give number answers the unit and scale the document prints, without changing the number.

    python -m lab.units answers.jsonl --model NAME

Numbers often lose their scale or unit ("$59,529" for "59,529 millions of Canadian dollars", "925-960" for
"925-960 MHz"), and the judge marks them wrong. Only answers that are numbers are re-read, using just their
cited pages. A reply is used only if it keeps exactly the same digits, so the value itself can't change.
Saved as a new file ending in _units.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .profiling import format_table, reset, snapshot, timed
from .read import call_server, final_json, render_jpeg

NUMBERISH = re.compile(r"^[\s$€£¥%.,:\-–/()+0-9a-zA-Z]*\d[\s$€£¥%.,:\-–/()+0-9a-zA-Z]*$")

PROMPT = """Question: {question}
Answer: {answer}

Find where this value appears on the page(s) above. Rewrite each answer item with the unit and scale the
document gives for it, for example currency, "million", "thousand", "%", "MHz", "kg", "days". Look at table
headers and column labels for the scale. Keep the number exactly as written. If the document gives no unit or
scale, keep the item unchanged. Don't add explanations.
Reply with JSON only: {{"answer": ["..."]}}"""


def is_number_answer(answer: list) -> bool:
    items = [str(a) for a in answer]
    return bool(items) and all(NUMBERISH.match(a) and sum(c.isdigit() for c in a) >= sum(c.isalpha() for c in a) for a in items)


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def accept(old: list, new: list) -> bool:
    """Same number of items, same digits in each, and only short additions."""
    if len(old) != len(new):
        return False
    for before, after in zip(old, new):
        before, after = str(before), str(after)
        if digits(before) != digits(after) or len(after) > len(before) + 40:
            return False
    return True


def add_units(row: dict, call: Callable[[list[dict]], dict], data: Path, max_side: int) -> dict:
    result = {**row, "unit_answer": row["answer"], "units_changed": False}
    if not is_number_answer(row["answer"]) or not row.get("citations"):
        return result
    content: list[dict] = []
    for citation in row["citations"][:3]:
        image = render_jpeg(data / "pdfs" / citation["file"], int(citation["page"]), max_side)
        content.append({"type": "text", "text": f"Page (file {citation['file']}, its page {citation['page']}):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    prompt = PROMPT.format(question=row["question"], answer=json.dumps(row["answer"], ensure_ascii=False))
    content.append({"type": "text", "text": prompt})
    with timed("units"):
        try:
            reply = call(content)
        except (urllib.error.URLError, OSError) as error:
            print(f"REQUEST FAILED {row['id']}: {error}", flush=True)
            return result
    parsed, _ = final_json(reply["choices"][0]["message"]["content"] or "")
    if parsed is None:
        return result
    new = parsed["answer"] if isinstance(parsed["answer"], list) else [parsed["answer"]]
    new = [str(item).strip() for item in new]
    if new != [str(a) for a in row["answer"]] and accept(row["answer"], new):
        result["answer"], result["units_changed"] = new, True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.answers.open(encoding="utf-8")]
    call = lambda content: call_server(args.base_url, args.model, content)
    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(lambda r: add_units(r, call, args.data, args.max_side), rows))
    path = args.answers.with_name(f"{args.answers.stem}_units.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    numbers = sum(is_number_answer(r["answer"]) for r in rows)
    print(f"Number answers {numbers}, changed {sum(o['units_changed'] for o in outputs)} of {len(rows)}; "
          f"seconds {time.perf_counter() - start:.0f}; saved {path}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
