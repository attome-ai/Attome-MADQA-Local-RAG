"""Learned page ranker: choose the reader's 9 pages from the text ranking, the ColQwen ranking and document context.

    python -m lab.pagerank            # train on train, report recall of the 9 chosen pages on dev (and 5-fold on train)

Labels are the gold pages of the 1,550 train questions, so nothing here is judged by the reader. Features use only
ranks and positions: rank in each ranking, the document's best rank and how many of its pages rank high, whether
the page is in the top-ranked document, its distance to the top-ranked page, and page number.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

DATA = Path("data")


def load(split: str):
    ledger = [json.loads(l) for l in (DATA / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    m = [e for e in ledger if e["name"] == "thin_ocr_visual" and e["split"] == split and e["limit"] is None][-1]
    text = {}
    for l in (DATA / "results" / "runs" / f"{m['key']}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        text[r["id"]] = [tuple(p) for p in r["ranked"]]
    vis = {}
    for l in (DATA / "results" / f"visual_{split}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        vis[r["id"]] = [tuple(p) for p in r["ranked"]]
    qs = [json.loads(l) for l in (DATA / "questions" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    return qs, text, vis


def features(tr: list, vr: list):
    pages = list(dict.fromkeys(tr + vr))
    trank = {p: i for i, p in enumerate(tr)}
    vrank = {p: i for i, p in enumerate(vr)}
    best_t, best_v, top20 = defaultdict(lambda: 200), defaultdict(lambda: 100), defaultdict(int)
    for p, i in trank.items():
        best_t[p[0]] = min(best_t[p[0]], i)
        top20[p[0]] += i < 20
    for p, i in vrank.items():
        best_v[p[0]] = min(best_v[p[0]], i)
    top_t, top_v = tr[0], vr[0] if vr else tr[0]
    rows = []
    for p in pages:
        t, v = trank.get(p, 150), vrank.get(p, 75)
        same_t = p[0] == top_t[0]
        rows.append([
            t, v, min(t, v * 2), 1 / (t + 3) + 1 / (v + 3), best_t[p[0]], best_v[p[0]], top20[p[0]],
            same_t, p[0] == top_v[0], abs(p[1] - top_t[1]) if same_t else 99,
            min([abs(p[1] - q[1]) for q in tr[:3] if q[0] == p[0]] or [99]),
            min([abs(p[1] - q[1]) for q in vr[:3] if q[0] == p[0]] or [99]), p[1],
        ])
    return pages, np.array(rows, dtype=float)


def build(split: str):
    qs, text, vis = load(split)
    out = []
    for q in qs:
        if q["id"] not in text or not q.get("evidence"):
            continue
        gold = {(e["document"], int(e["page"])) for e in q["evidence"]}
        pages, x = features(text[q["id"]], vis.get(q["id"], []))
        out.append((q["id"], gold, text[q["id"]], pages, x, np.array([p in gold for p in pages])))
    return out


def report(name: str, items, choose):
    stats = defaultdict(lambda: [0, 0])
    for qid, gold, tr, pages, x, y in items:
        kind = "single" if len(gold) == 1 else "multi"
        got = set(choose(qid, gold, tr, pages, x))
        stats[kind][0] += gold <= got
        stats[kind][1] += 1
    print(name, {k: f"{a}/{b}" for k, (a, b) in stats.items()})


def export(split: str, items, score) -> None:
    """Save the ranker's order as an experiment run named thin_ocr_ranker, so lab.answer can read it.
    Train rows use out-of-fold scores, so a train question is never ranked by a model that saw it."""
    ledger_path = DATA / "results" / "ledger.jsonl"
    ledger = [json.loads(l) for l in ledger_path.read_text(encoding="utf-8").splitlines()]
    base = [e for e in ledger if e["name"] == "thin_ocr_visual" and e["split"] == split and e["limit"] is None][-1]
    key = "ranker_" + split
    path = DATA / "results" / "runs" / f"{key}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            s = score(item)
            order = [item[3][j] for j in np.argsort(-s)]
            handle.write(json.dumps({"id": item[0], "ranked": [list(p) for p in order]}) + chr(10))
    entry = {**base, "key": key, "name": "thin_ocr_ranker", "hypothesis": "Learned page ranker picks the reader's pages."}
    ledger = [e for e in ledger if e["key"] != key]
    with ledger_path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(e) + chr(10) for e in ledger + [entry])


def main() -> None:
    train, dev = build("train"), build("dev")
    model = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=0)  # noqa: E731
    base = lambda qid, gold, tr, pages, x: tr[:9]  # noqa: E731
    X = np.vstack([i[4] for i in train])
    Y = np.concatenate([i[5] for i in train])
    G = np.concatenate([[n] * len(i[3]) for n, i in enumerate(train)])
    scores = np.zeros(len(Y))
    for a, b in GroupKFold(5).split(X, Y, G):
        scores[b] = model().fit(X[a], Y[a]).predict_proba(X[b])[:, 1]
    cut = np.cumsum([0] + [len(i[3]) for i in train])
    cv = {train[n][0]: scores[cut[n] : cut[n + 1]] for n in range(len(train))}
    pick = lambda s, pages, k=9: [pages[j] for j in np.argsort(-s)[:k]]  # noqa: E731
    report("train current top9    ", train, base)
    report("train ranker (5-fold) ", train, lambda qid, gold, tr, pages, x: pick(cv[qid], pages))
    final = model().fit(X, Y)
    report("dev   current top9    ", dev, base)
    report("dev   ranker          ", dev, lambda qid, gold, tr, pages, x: pick(final.predict_proba(x)[:, 1], pages))
    export("dev", dev, lambda it: final.predict_proba(it[4])[:, 1])
    export("train", train, lambda it: cv[it[0]])
    report("dev   ranker top12    ", dev, lambda qid, gold, tr, pages, x: pick(final.predict_proba(x)[:, 1], pages, 12))


if __name__ == "__main__":
    main()
