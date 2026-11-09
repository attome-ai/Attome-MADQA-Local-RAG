"""Maths in code: the model reads values off the pages, and Python does the arithmetic.

    python -m lab.calc data/results/answers_x.jsonl experiments/thin_ocr_llm_select.json --model NAME

The model says which operation a question needs. Most questions need none and keep their draft.
Nothing the model writes is ever executed: only a fixed list of operations can run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from .answer import build_content, parse_reply, saved_rankings
from .profiling import format_table, reset, snapshot, timed
from .read import call_server, final_json
from .run import read_jsonl, sha256

OPERATIONS = ("sum", "difference", "count", "highest", "lowest", "earliest", "latest", "years_between")

CALC_PROMPT = """Does this question need arithmetic, counting, or comparing values? If so, don't do
the calculation yourself: copy each value exactly as printed on the pages, with a short label.
Operations: sum, difference (first minus second), count (one item per thing counted), highest,
lowest, earliest, latest (for these four, labels are what the answer should name),
years_between (two dates). Use "none" if the question only needs looking something up.
Reply with JSON only:
{{"operation": "none", "values": [{{"label": "...", "value": "...", "page": 1}}], "unit": ""}}

Question: {question}"""

MONTHS = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y")
NUMERIC_DATES = ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d")


def parse_number(text: object) -> float | None:
    """Read a printed number, including $, commas, percent signs and accounting brackets."""
    raw = str(text).strip()
    negative = raw.startswith("(") and raw.endswith(")") or raw.startswith("-")
    match = re.search(r"\d[\d,]*(?:\.\d+)?|\.\d+", raw)
    if not match:
        return None
    # A value with other digits or words beyond a unit is ambiguous, such as "3 of 5".
    remainder = (raw[: match.start()] + raw[match.end() :]).lower()
    if re.search(r"\d", remainder):
        return None
    value = float(match.group().replace(",", ""))
    if re.search(r"\b(million|mln)\b", remainder):
        value *= 1_000_000
    elif re.search(r"\b(billion|bln)\b", remainder):
        value *= 1_000_000_000
    elif re.search(r"\bthousand\b", remainder):
        value *= 1_000
    return -value if negative else value


def parse_date(text: object) -> date | None:
    raw = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", str(text).strip().rstrip("."))
    raw = re.sub(r"\s+", " ", raw)
    for pattern in MONTHS + NUMERIC_DATES:
        try:
            return datetime.strptime(raw, pattern).date()
        except ValueError:
            continue
    return None


def format_number(value: float, values: list[str], unit: str) -> str:
    decimals = max((len(v.split(".")[-1]) for v in values if "." in re.sub(r"[^\d.]", "", v)), default=0)
    decimals = min(decimals, 4)
    body = f"{value:,.{decimals}f}" if decimals else f"{round(value):,}"
    prefix = "$" if any("$" in v for v in values) else ""
    return f"{prefix}{body}{' ' + unit.strip() if unit.strip() else ''}"


def compute(operation: str, values: list[dict], unit: str = "") -> str | None:
    """Return the computed answer, or None whenever it can't be worked out with confidence."""
    if operation not in OPERATIONS or not values:
        return None
    raw = [str(v.get("value", "")) for v in values]
    if operation == "count":
        return f"{len(values)}{' ' + unit.strip() if unit.strip() else ''}"
    if operation in ("earliest", "latest"):
        dates = [parse_date(r) for r in raw]
        if len(dates) < 2 or None in dates:
            return None
        pick = min(dates) if operation == "earliest" else max(dates)
        if dates.count(pick) > 1:
            return None
        return str(values[dates.index(pick)].get("label", "")) or None
    if operation == "years_between":
        dates = [parse_date(r) for r in raw]
        if len(dates) != 2 or None in dates:
            return None
        early, late = sorted(dates)
        return str(late.year - early.year - ((late.month, late.day) < (early.month, early.day)))
    numbers = [parse_number(r) for r in raw]
    if None in numbers:
        return None
    if operation == "sum":
        return format_number(sum(numbers), raw, unit) if len(numbers) >= 2 else None
    if operation == "difference":
        return format_number(numbers[0] - numbers[1], raw, unit) if len(numbers) == 2 else None
    if len(numbers) < 2:
        return None
    pick = max(numbers) if operation == "highest" else min(numbers)
    if numbers.count(pick) > 1:
        return None
    return str(values[numbers.index(pick)].get("label", "")) or None


def apply_calc(draft: dict, text: str, pages: list) -> dict:
    """Replace the draft only when a real operation gives a computed answer."""
    result = {**draft, "calc_changed": False}
    data, _ = final_json(text.replace('"operation"', '"answer": [], "operation"', 1))
    if data is None:
        return result
    values = data.get("values")
    operation = data.get("operation")
    result["calc_operation"] = operation
    if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
        return result
    computed = compute(str(operation), values, str(data.get("unit") or ""))
    if computed is None:
        return result
    result["calc_answer"] = computed
    if [computed] != draft["answer"]:
        result["draft_before_calc"] = draft["answer"]
        result["answer"], result["calc_changed"] = [computed], True
        labels = list(dict.fromkeys(v.get("page") for v in values))
        _, citations = parse_reply(json.dumps({"answer": [computed], "pages": labels}), pages)
        if citations:
            result["citations"] = citations
    return result


def calc_questions(
    drafts: list[dict],
    ranked: dict[str, list],
    call: Callable[[list[dict]], dict],
    data: Path,
    top_k: int,
    max_side: int | None,
    workers: int = 6,
) -> list[dict]:
    def one(draft: dict) -> dict:
        pages = ranked[draft["id"]][:top_k]
        if not pages:
            return {**draft, "calc_changed": False}
        content = build_content(draft["question"], pages, data, "", max_side, True)
        content[-1] = {"type": "text", "text": CALC_PROMPT.format(question=draft["question"])}
        with timed("calc_total"):
            try:
                reply = call(content)
            except (urllib.error.URLError, OSError) as error:
                print(f"REQUEST FAILED {draft['id']}: {error}", flush=True)
                return {**draft, "calc_changed": False}
        return apply_calc(draft, reply["choices"][0]["message"]["content"] or "", pages)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, drafts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    reset()
    start = time.perf_counter()
    outputs = calc_questions(
        read_jsonl(args.answers),
        ranked,
        lambda content: call_server(args.base_url, args.model, content),
        args.data,
        args.top_k,
        args.max_side,
        args.workers,
    )
    seconds = time.perf_counter() - start
    path = args.answers.with_name(f"{args.answers.stem}_calc.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    used = sum(bool(o.get("calc_answer")) for o in outputs)
    print(path)
    print(f"Calculated {used}, changed {sum(o['calc_changed'] for o in outputs)} of {len(outputs)}; "
          f"questions per second {len(outputs) / seconds:.3f}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
