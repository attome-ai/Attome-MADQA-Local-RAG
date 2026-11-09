"""Does a question need every matching item from a document (a collect question) or one fact (a lookup)?

Asked of the local model once per question, text only. Used to decide which questions get the deep read.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .read import call_server

PROMPT = """Question about a document: {question}

Does answering this need EVERY matching item, row or entry from the document (a collection that could run over
several pages, such as all the names, all the years meeting a condition, all the entries of a type), or just one
specific fact, value or short answer?
Reply with JSON only: {{"collect": true}} or {{"collect": false}}"""


def classify(questions: list[str], model: str, base_url: str = "http://localhost:8000/v1", workers: int = 8, cache: Path | None = None) -> dict[str, bool]:
    known = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    todo = [q for q in dict.fromkeys(questions) if q not in known]

    def one(q: str) -> tuple[str, bool]:
        try:
            reply = call_server(base_url, model, [{"type": "text", "text": PROMPT.format(question=q)}])
            text = (reply["choices"][0]["message"]["content"] or "").rsplit("</think>", 1)[-1]
            return q, '"collect": true' in text.replace('"collect":true', '"collect": true')
        except Exception:
            return q, False

    with ThreadPoolExecutor(workers) as pool:
        known.update(dict(pool.map(one, todo)))
    if cache:
        cache.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
    return known
