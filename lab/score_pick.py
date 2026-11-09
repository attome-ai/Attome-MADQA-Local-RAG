"""Score-and-pick (T34): judge each candidate answer on its own, then keep the most likely one.

    python -m lab.score_pick out.jsonl first.jsonl second.jsonl [more.jsonl] --model NAME

The adjudicator compares all candidates in one prompt and tends to follow the majority or the first one. Here
each distinct candidate is checked separately against the same pages (every page any candidate cites), and the
model's probability of "yes" to "is this answer correct?" is its score. The highest score wins. One output token
per candidate; the page images are the same each time, so the server's cache makes repeat checks cheap.
Unanimous questions keep the shared answer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adjudicate import distinct_candidates, evidence_pages
from .profiling import format_table, reset, snapshot, timed
from .read import MAX_IMAGES, render_jpeg
from .run import read_jsonl

CHECK = """Question: {question}
Proposed answer: {answer}

Using only the pages above, is the proposed answer correct and complete for exactly what the question asks
(the right row, person, year and unit, with nothing missing or extra)? Reply with one word: yes or no."""


def yes_probability(base_url: str, model: str, content: list[dict]) -> float:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with timed("score_call"), urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    top = reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no else 0.0


def page_blocks(pages: list, data: Path, max_side: int) -> list[dict]:
    content: list[dict] = []
    for number, (document, page) in enumerate(pages, 1):
        image = render_jpeg(data / "pdfs" / document, page, max_side)
        content.append({"type": "text", "text": f"Page {number} (file {document}, its page {page}):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    return content


def pick(rows: list[dict], score, data: Path, max_side: int, limit: int = MAX_IMAGES, own: bool = False) -> dict:
    """own=True checks each candidate only against the pages it cites (up to 3), so its evidence is never crowded out."""
    candidates = distinct_candidates(rows)
    if len(candidates) < 2:
        return {**(candidates[0] if candidates else rows[0]), "scored": False}
    pages = evidence_pages(candidates, limit)
    if not pages:
        return {**candidates[0], "scored": False}
    shared = None if own else page_blocks(pages, data, max_side)
    scores = []
    for candidate in candidates:
        prompt = CHECK.format(question=rows[0]["question"], answer=json.dumps(candidate["answer"], ensure_ascii=False))
        if own:
            cited = [(c["file"], int(c["page"])) for c in candidate.get("citations", [])][:3]
            if not cited:
                scores.append(-1.0)
                continue
            content = page_blocks(list(dict.fromkeys(cited)), data, max_side)
        else:
            content = shared
        try:
            scores.append(score(content + [{"type": "text", "text": prompt}]))
        except OSError:
            scores.append(-1.0)
    best = max(range(len(candidates)), key=lambda i: scores[i])
    return {
        **candidates[best],
        "scored": True,
        "scores": [[c["answer"], round(s, 3)] for c, s in zip(candidates, scores)],
        "iterations": max(int(r.get("iterations", 1)) for r in rows) + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--own", action="store_true", help="check each candidate only against its own cited pages")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    tables = [{row["id"]: row for row in read_jsonl(path)} for path in args.files]
    ids = [i for i in tables[0] if all(i in t for t in tables)]
    score = lambda content: yes_probability(args.base_url, args.model, content)
    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        outputs = list(pool.map(lambda i: pick([t[i] for t in tables], score, args.data, args.max_side, own=args.own), ids))
    with args.out.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(f"Scored {sum(o['scored'] for o in outputs)} of {len(outputs)}; seconds {time.perf_counter() - start:.0f}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
