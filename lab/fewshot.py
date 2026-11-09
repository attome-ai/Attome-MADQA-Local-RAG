"""Dynamic few-shot: show the reader how similar train questions were answered (question and gold answer only).

The pool is the train split, so dev and test questions never contribute answers. A question never sees itself.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

STOP = set("the a an of in on for to and or is are was were what which who whom how many much does did do by with at as from that this their its it be".split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


@lru_cache(maxsize=1)
def pool(path: str = "data/questions/train.jsonl"):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    docs = [(r["question"], r["answer_variants"][0] if r.get("answer_variants") else [], Counter(words(r["question"]))) for r in rows]
    df = Counter(w for _, _, c in docs for w in c)
    return docs, df, len(docs)


def examples(question: str, k: int = 4) -> str:
    docs, df, n = pool()
    q = Counter(words(question))
    scored = []
    for text, answer, c in docs:
        if text == question or not answer:
            continue
        s = sum(math.log(n / df[w]) * min(q[w], c[w]) for w in q if w in c)
        scored.append((s / math.sqrt(1 + sum(c.values())), text, answer))
    scored.sort(reverse=True)
    lines = [f"Question: {t}\nAnswer: {json.dumps(a, ensure_ascii=False)}" for _, t, a in scored[:k]]
    return (
        "Examples of how similar questions about other documents were answered. They show the expected form and "
        "amount of detail only; their facts do not apply to this question.\n\n" + "\n\n".join(lines)
    )
