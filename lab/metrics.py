"""Retrieval metrics. Hop types match MADQA's eval/evaluate.py."""

from __future__ import annotations

from .profiling import profiled


def gold_pages(question: dict) -> set[tuple[str, int]]:
    return {(e["document"], int(e["page"])) for e in question["evidence"]}


def hop_type(question: dict) -> str:
    gold = gold_pages(question)
    if len({doc for doc, _ in gold}) > 1:
        return "cross_doc"
    return "cross_page" if len(gold) > 1 else "single"


def _summarize(rows: list[tuple[dict, list[tuple[str, int]]]], ks: list[int]) -> dict:
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    for k in ks:
        recall = all_found = doc_recall = 0.0
        for question, ranked in rows:
            gold = gold_pages(question)
            top = set(ranked[:k])
            hits = len(gold & top)
            recall += hits / len(gold)
            all_found += hits == len(gold)
            gold_docs = {doc for doc, _ in gold}
            doc_recall += len(gold_docs & {doc for doc, _ in top}) / len(gold_docs)
        out[f"page_recall@{k}"] = round(100 * recall / len(rows), 2)
        out[f"all_evidence@{k}"] = round(100 * all_found / len(rows), 2)
        out[f"doc_recall@{k}"] = round(100 * doc_recall / len(rows), 2)
    return out


@profiled
def evaluate(rows: list[tuple[dict, list[tuple[str, int]]]], ks: list[int]) -> dict:
    """rows holds (question, ranked pages). Questions without evidence are skipped."""
    rows = [row for row in rows if row[0].get("evidence")]
    result = {"overall": _summarize(rows, ks)}
    for hop in ("single", "cross_page", "cross_doc"):
        result[hop] = _summarize([r for r in rows if hop_type(r[0]) == hop], ks)
    return result
