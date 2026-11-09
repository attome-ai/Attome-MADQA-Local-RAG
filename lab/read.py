"""Experiment 4: how well can a vision model answer when handed the correct pages?

    python -m lab.read --model Qwen/Qwen3.5-9B --mode image --split dev --limit 20

Talks to any OpenAI-compatible server. Scoring uses MADQA's own ANLS* code. Accuracy here means
ANLS* >= 0.5, which is stricter than the leaderboard's LLM-judged accuracy, so treat scores as
comparisons between readers rather than leaderboard predictions.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .metrics import hop_type
from .profiling import format_table, profiled, reset, snapshot, timed
from .run import read_jsonl, select_questions

PROMPT = """Answer the question using only the document pages provided.
Give the shortest exact answer, copied from the page where possible, with no explanation.
If the question has several answers, give each as a separate list item.
Reply with JSON only: {"answer": ["..."]}

Question: {question}"""
RULES_V2 = """Rules:
- Yes/no question: answer exactly "Yes" or "No".
- Give only the value asked for, never a sentence. Copy words and numbers exactly as printed.
- Several answers: put each item in its own list entry, never joined by commas.
- Read tables, stamps, checkboxes and handwriting carefully. Do any needed counting or arithmetic.
- Always give your best answer. Never leave the answer empty."""
RULES_V3 = """Rules:
- Yes/no question: answer exactly "Yes" or "No", even if the page words it differently.
- Comparison question: answer with the name of the winner, not the values.
- Give only what was asked, never a sentence, but keep units and qualifiers: "2,400 cwt", "$51.1 mln", "over 250".
- Copy names, codes and numbers character by character as printed.
- Tables and forms: find the exact row and column before answering, and check the year or period in the heading matches the question.
- Total or count: add up every relevant item and give one final number.
- List question: include every item, each in its own entry, without repeats.
- If the pages truly lack the answer, give your best supported answer rather than leaving it empty."""
RULES_WORKSPACE = RULES_V3 + """
- Before answering, fill "facts" with each relevant value you can see: what it is, the value exactly as
  printed, and its page label. Include the row and column or the heading it came from. Then answer only
  from those facts. Add "facts" to your JSON, for example:
  {"facts": [{"item": "...", "value": "...", "source": "...", "page": 1}], "answer": ["..."], "pages": [1]}"""
RULES_V3_ID = RULES_V3 + """
- "The number of", "the number assigned to" or "what is the number" followed by a named item (certificate, case, form, section, step, item, policy) asks for that item's own identifier as printed, not a quantity. Only "how many" or a count of things asks for a quantity."""
RULES = {"v1": "", "v2": RULES_V2, "v3": RULES_V3, "workspace": RULES_WORKSPACE, "v3_id": RULES_V3_ID}
MAX_SIDE = 1568
# Matches the vLLM server's per-prompt image limit.
MAX_IMAGES = 9  # 9 pages at 2,048 px fill the 32k context; vLLM is started with the same limit
# PDFium is not thread-safe, so only one thread may render at a time.
_render_lock = threading.Lock()


# Set MADQA_TEMPERATURE to sample varied reads (default 0, repeatable). Used for extra adjudication candidates.
TEMPERATURE = float(os.environ.get("MADQA_TEMPERATURE", "0"))
# Set MADQA_TRIM=1 to crop blank page margins at the same scale: same text sharpness, about 15% fewer image tokens.
TRIM = os.environ.get("MADQA_TRIM") == "1"


def trim_margins(image, pad: int = 16, threshold: int = 25):
    """Crop to the non-white content plus a small pad. Blank pages are left as they are."""
    from PIL import ImageOps

    box = ImageOps.invert(image.convert("L")).point(lambda v: 255 if v > threshold else 0).getbbox()
    if not box:
        return image
    width, height = image.size
    return image.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(width, box[2] + pad), min(height, box[3] + pad)))


@profiled
def render_jpeg(path: Path, page: int, max_side: int | None = None) -> str:
    import pypdfium2

    with _render_lock:
        pdf = pypdfium2.PdfDocument(path)
        try:
            pdf_page = pdf[page - 1]
            scale = (max_side or MAX_SIDE) / max(pdf_page.get_size())
            image = pdf_page.render(scale=scale).to_pil().convert("RGB")
        finally:
            pdf.close()
    if TRIM:
        image = trim_margins(image)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def final_json(text: str) -> tuple[dict | None, str]:
    """Return the final JSON object and the text after any thinking.

    With thinking on, some templates open the think block themselves, so only the closing tag
    appears in the output. Everything before the last closing tag is reasoning.
    """
    text = text.rsplit("</think>", 1)[-1].strip()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return data, text
    return None, text


def parse_answer(text: str) -> list[str]:
    """Pull the answer list from model output, tolerating fences and thinking."""
    data, rest = final_json(text)
    if data is None:
        return [rest] if rest else []
    answer = data["answer"]
    return [str(item) for item in answer] if isinstance(answer, list) else [str(answer)]


def with_neighbors(pages: list[tuple[str, int]], window: int, known: set) -> list[tuple[str, int]]:
    """Keep the given pages first, then add adjacent pages that exist, nearest first."""
    result = list(pages)
    for distance in range(1, window + 1):
        for document, page in pages:
            for near in ((document, page + distance), (document, page - distance)):
                if near in known and near not in result:
                    result.append(near)
    return result


def build_content(
    question: dict, data: Path, mode: str, text: dict, rules: str = "", window: int = 0, known: set = frozenset(), max_side: int | None = None
) -> list[dict]:
    pages = sorted({(e["document"], int(e["page"])) for e in question["evidence"]})
    pages = with_neighbors(pages, window, known)[:MAX_IMAGES]
    prompt = PROMPT.replace("{question}", question["question"])
    if rules:
        prompt = f"{rules}\n\n{prompt}"
    content: list[dict] = []
    if mode in ("image", "image+text"):
        for document, page in pages:
            image = render_jpeg(data / "pdfs" / document, page, max_side)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    if mode in ("text", "image+text"):
        extracted = "\n\n".join(f"[{d} page {p}]\n{text.get((d, p), '')}" for d, p in pages)
        prompt = f"Extracted page text, which may contain errors:\n{extracted}\n\n{prompt}"
    content.append({"type": "text", "text": prompt})
    return content


def collect_stream(lines) -> dict:
    """Rebuild a normal chat reply from server-sent stream lines."""
    parts, usage = [], {}
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            parts.append(choice.get("delta", {}).get("content") or "")
    return {"choices": [{"message": {"content": "".join(parts)}}], "usage": usage}


def call_server(base_url: str, model: str, content: list[dict], thinking: bool = False) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": TEMPERATURE,
        "max_tokens": 10000 if thinking else 2048,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with timed("llm_call"), urllib.request.urlopen(request, timeout=600) as response:
        return collect_stream(response)


def run_reader(
    questions: list[dict],
    data: Path,
    mode: str,
    text: dict,
    call: Callable[[list[dict]], dict],
    workers: int = 8,
    rules: str = "",
    window: int = 0,
    known: set = frozenset(),
    max_side: int | None = None,
) -> list[dict]:
    def one(question: dict) -> dict:
        with timed("question_total"):
            reply = call(build_content(question, data, mode, text, rules, window, known, max_side))
        return {
            "id": question["id"],
            "question": question["question"],
            "answer": parse_answer(reply["choices"][0]["message"]["content"] or ""),
            "usage": reply.get("usage", {}),
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, questions))


def score(questions: list[dict], outputs: list[dict]) -> dict:
    from .madqa_metrics import anls_star

    rows = [(q, anls_star(o["answer"], q["answer_variants"])) for q, o in zip(questions, outputs)]

    def summary(items: list) -> dict:
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "anls": round(100 * sum(s for _, s in items) / len(items), 2),
            "accuracy": round(100 * sum(s >= 0.5 for _, s in items) / len(items), 2),
        }

    result = {"overall": summary(rows)}
    for hop in ("single", "cross_page", "cross_doc"):
        result[hop] = summary([row for row in rows if hop_type(row[0]) == hop])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["image", "text", "image+text"], default="image")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--pages", default="hybrid_rapidocr")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--rules", choices=["v1", "v2", "v3"], default="v1")
    parser.add_argument("--max-side", type=int, help="longest page image side in pixels")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--ids", type=Path, help="only run the questions in this id-set file")
    parser.add_argument("--provider", choices=["local", "commandcode"], default="local")
    parser.add_argument("--neighbors", type=int, default=0, help="also show pages this close to each given page")
    args = parser.parse_args()

    available = {path.name for path in (args.data / "pdfs").glob("*.pdf")}
    questions = select_questions(read_jsonl(args.data / "questions" / f"{args.split}.jsonl"), args.ids)
    usable = [q for q in questions if all(e["document"] in available for e in q["evidence"])]
    print(f"Questions with all evidence downloaded: {len(usable)} of {len(questions)}")
    usable = usable[: args.limit]
    text = {}
    if args.mode != "image":
        rows = read_jsonl(args.data / "pages" / f"{args.pages}.jsonl")
        text = {(row["document"], row["page"]): row["text"] for row in rows}

    reset()
    start = time.perf_counter()
    rules = RULES[args.rules]
    if args.provider == "commandcode":
        from .commandcode import ClaudeReader

        call = ClaudeReader(args.model)
    else:
        call = lambda content: call_server(args.base_url, args.model, content, args.thinking)
    known = set()
    if args.neighbors:
        rows = read_jsonl(args.data / "pages" / "pdf_text.jsonl")
        known = {(row["document"], row["page"]) for row in rows}
    outputs = run_reader(usable, args.data, args.mode, text, call, args.workers, rules, args.neighbors, known, args.max_side)
    seconds = time.perf_counter() - start
    tokens = [o["usage"].get("completion_tokens", 0) for o in outputs]
    entry = {
        "model": args.model,
        "mode": args.mode,
        "split": args.split,
        "limit": args.limit,
        "workers": args.workers,
        "rules": args.rules,
        "thinking": args.thinking,
        "neighbors": args.neighbors,
        "max_side": args.max_side,
        "ids": args.ids.stem if args.ids else None,
        "metrics": score(usable, outputs),
        "questions_per_second": round(len(usable) / seconds, 3),
        "output_tokens_per_second": round(sum(tokens) / seconds, 1),
        "profile": snapshot(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results = args.data / "results"
    results.mkdir(parents=True, exist_ok=True)
    with (results / "reader_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    variant = f"_{args.rules}" + ("_think" if args.thinking else "") if args.rules != "v1" or args.thinking else ""
    name = f"{args.model.replace('/', '_')}_{args.mode.replace('+', '_')}_{args.split}{variant}" + (f"_{args.ids.stem}" if args.ids else "") + (f"_near{args.neighbors}" if args.neighbors else "") + (f"_px{args.max_side}" if args.max_side else "")
    with (results / f"reader_{name}.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(json.dumps(entry["metrics"], indent=2))
    print(f"Questions per second: {entry['questions_per_second']}")
    print(format_table(entry["profile"]))


if __name__ == "__main__":
    main()
