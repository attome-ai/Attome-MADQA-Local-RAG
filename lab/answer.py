"""End-to-end run: retrieve pages, read them with a vision model, answer with citations.

    python -m lab.answer experiments/bm25_hybrid_ocr.json --model NAME --top-k 5 --split dev

Output rows use the leaderboard's fields, so the format carries over to a later test run.
Accuracy here means ANLS* >= 0.5, without the leaderboard's LLM judge.
Scores are reported for all questions and for those whose evidence PDFs were all downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .metrics import hop_type
from .profiling import format_table, profiled, reset, snapshot, timed
from .read import TEMPERATURE, TRIM, RULES, call_server, final_json, render_jpeg
from .run import read_jsonl, select_questions, sha256
from .stages import PageIndex, run_pipeline

PROMPT = """You are given {count} document pages, labelled Page 1 to Page {count} in order.
Answer the question using only these pages. Give the shortest exact answer, copied from the
page where possible, with no explanation. If there are several answers, give each as a separate
list item. List only the page labels your answer actually relies on.
Reply with JSON only: {{"answer": ["..."], "pages": [1]}}

Question: {question}"""


def parse_reply(text: str, retrieved: list[tuple[str, int]]) -> tuple[list[str], list[dict]]:
    """Return answers and citations, ignoring page labels that were never shown."""
    data, rest = final_json(text)
    if data is None:
        return ([rest] if rest else []), []
    answer = data["answer"]
    if not isinstance(answer, list):
        answer = [answer]
    labels = data.get("pages", [])
    if not isinstance(labels, list):
        labels = [labels]
    citations, seen = [], set()
    for label in labels:
        if isinstance(label, int) and 1 <= label <= len(retrieved) and label not in seen:
            seen.add(label)
            document, page = retrieved[label - 1]
            citations.append({"file": document, "page": page})
    return [str(item) for item in answer], citations


def build_content(
    question: str, pages: list[tuple[str, int]], data: Path, rules: str = "", max_side: int | None = None, doc_labels: bool = False
) -> list[dict]:
    content: list[dict] = []
    for number, (document, page) in enumerate(pages, 1):
        image = render_jpeg(data / "pdfs" / document, page, max_side)
        label = f"Page {number} (file {document}, its page {page}):" if doc_labels else f"Page {number}:"
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
    prompt = PROMPT.format(count=len(pages), question=question)
    if rules:
        prompt = f"{rules}\n\n{prompt}"
    content.append({"type": "text", "text": prompt})
    return content


def answer_questions(
    questions: list[dict],
    retrieve: Callable[[str], list[tuple[str, int]]],
    call: Callable[[list[dict]], dict],
    data: Path,
    top_k: int,
    workers: int = 8,
    rules: str = "",
    max_side: int | None = None,
    doc_labels: bool = False,
) -> list[dict]:
    # SQLite connections stay on one thread, so retrieval finishes before reading starts.
    retrieved = [retrieve(q["question"])[:top_k] for q in questions]

    def one(item: tuple[dict, list[tuple[str, int]]]) -> dict:
        question, pages = item
        with timed("question_total"):
            answer, citations = [], []
            usage = {}
            if pages:
                try:
                    instructions = rules(question["question"]) if callable(rules) else rules
                    reply = call(build_content(question["question"], pages, data, instructions, max_side, doc_labels))
                except urllib.error.HTTPError as error:
                    # One oversized request should not lose the whole run. It scores as empty.
                    print(f"REQUEST FAILED {question['id']}: HTTP {error.code}", flush=True)
                else:
                    usage = reply.get("usage", {})
                    answer, citations = parse_reply(reply["choices"][0]["message"]["content"] or "", pages)
        return {
            "question": question["question"],
            "answer": answer,
            "citations": citations,
            "iterations": 1,
            "id": question["id"],
            "usage": usage,
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, zip(questions, retrieved)))


@profiled
def score(questions: list[dict], outputs: list[dict]) -> dict:
    from .madqa_metrics import anls_star, citation_f1

    rows = []
    for question, output in zip(questions, outputs):
        rows.append(
            (
                question,
                anls_star(output["answer"], question["answer_variants"]),
                citation_f1(output["citations"], question["evidence"], level="page")["f1"],
                citation_f1(output["citations"], question["evidence"], level="document")["f1"],
            )
        )

    def summary(items: list) -> dict:
        if not items:
            return {"n": 0}
        count = len(items)
        return {
            "n": count,
            "accuracy": round(100 * sum(r[1] >= 0.5 for r in items) / count, 2),
            "anls": round(100 * sum(r[1] for r in items) / count, 2),
            "page_f1": round(100 * sum(r[2] for r in items) / count, 2),
            "doc_f1": round(100 * sum(r[3] for r in items) / count, 2),
        }

    result = {"overall": summary(rows)}
    for hop in ("single", "cross_page", "cross_doc"):
        result[hop] = summary([r for r in rows if hop_type(r[0]) == hop])
    return result


def rules_choice(name: str):
    """A fixed prompt, or a function choosing a specialist prompt per question."""
    if name.endswith("_fs"):
        from .fewshot import examples

        base = RULES[name[:-3]]
        return lambda question: base + chr(10) * 2 + examples(question)
    if name.startswith("router"):
        from .specialists import specialist_rules

        base = RULES["workspace"] if name.endswith("workspace") else RULES["v3"]
        return lambda question: specialist_rules(question, base)
    return RULES[name]


def saved_rankings(data: Path, name: str, split: str, pages_hash: str) -> dict[str, list[tuple[str, int]]]:
    """Load rankings from the latest full run of an experiment on the current page text."""
    matches = [
        e
        for e in read_jsonl(data / "results" / "ledger.jsonl")
        if e["name"] == name
        and e["split"] == split
        and e["limit"] is None
        and e.get("pages_hash") == pages_hash
    ]
    if not matches:
        raise ValueError(f"Run experiment {name} on {split} with the current page text first")
    rows = read_jsonl(data / "results" / "runs" / f"{matches[-1]['key']}.jsonl")
    return {row["id"]: [(doc, page) for doc, page in row["ranked"]] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--rules", choices=["v1", "v2", "v3", "workspace", "router", "router_workspace", "v3_fs", "workspace_fs", "v3_id"], default="v1")
    parser.add_argument("--max-side", type=int, help="longest page image side in pixels")
    parser.add_argument("--doc-labels", action="store_true", help="tell the reader each page's file and number")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--ids", type=Path, help="only run the questions in this id-set file")
    parser.add_argument("--offset", type=int, default=0, help="skip this many ranked pages, to read deeper in the ranking")
    parser.add_argument("--provider", choices=["local", "commandcode", "chat"], default="local")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    index_path = args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite"
    pages = read_jsonl(pages_path) if not index_path.exists() else None
    questions = select_questions(read_jsonl(args.data / "questions" / f"{args.split}.jsonl"), args.ids)[: args.limit]

    reset()
    index = PageIndex(index_path, pages)
    index.pages_path = pages_path
    if any(step["stage"] == "rerank" for step in config["pipeline"]):
        # The reranker runs in Docker, so reuse the rankings its experiment already saved.
        ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
        by_text = {q["question"]: ranked[q["id"]][args.offset :] for q in questions}
        retrieve = by_text.__getitem__
    else:
        retrieve = lambda text: run_pipeline(text, config["pipeline"], index)
    if args.provider == "commandcode":
        from .commandcode import ClaudeReader

        call = ClaudeReader(args.model)
    elif args.provider == "chat":
        from .commandcode import ChatReader

        call = ChatReader(args.model)
    else:
        call = lambda content: call_server(args.base_url, args.model, content, args.thinking)
    start = time.perf_counter()
    outputs = answer_questions(
        questions,
        retrieve,
        call,
        args.data,
        args.top_k,
        args.workers,
        rules_choice(args.rules),
        args.max_side,
        args.doc_labels,
    )
    seconds = time.perf_counter() - start

    available = {path.name for path in (args.data / "pdfs").glob("*.pdf")}
    usable = [
        i for i, q in enumerate(questions) if all(e["document"] in available for e in q["evidence"])
    ]
    metrics = {
        "all_questions": score(questions, outputs),
        "evidence_downloaded": score([questions[i] for i in usable], [outputs[i] for i in usable]),
    }
    entry = {
        "name": config["name"],
        "model": args.model,
        "top_k": args.top_k,
        "split": args.split,
        "limit": args.limit,
        "workers": args.workers,
        "rules": args.rules,
        "max_side": args.max_side,
        "doc_labels": args.doc_labels,
        "thinking": args.thinking,
        "ids": args.ids.stem if args.ids else None,
        "metrics": metrics,
        "questions_per_second": round(len(questions) / seconds, 3),
        "profile": snapshot(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results = args.data / "results"
    with (results / "answer_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    variant = args.rules + ("_think" if args.thinking else "") + (f"_px{args.max_side}" if args.max_side else "") + ("_labels" if args.doc_labels else "") + ("_trim" if TRIM else "") + (f"_off{args.offset}" if args.offset else "") + (f"_t{TEMPERATURE:g}_{os.environ.get('MADQA_SAMPLE', '0')}" if TEMPERATURE else "")
    path = results / f"answers_{config['name']}_{args.model.replace('/', '_')}_k{args.top_k}_{args.split}_{variant}{'_' + args.ids.stem if args.ids else ''}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    print(json.dumps(metrics, indent=2))
    print(f"Questions per second: {entry['questions_per_second']}")
    print(format_table(entry["profile"]))


if __name__ == "__main__":
    main()
