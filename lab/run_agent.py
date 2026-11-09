"""Run the agent on real questions. Reranking needs torch, so this runs inside Docker:

    python -m lab.run_agent experiments/thin_ocr_neighbors.json --model NAME --split train \
        --ids experiments/train_check.json --rules v3 --max-side 2048

Rows match lab.answer's output, so lab.judge and lab.submit can read them. All search work
stays on one thread, because the SQLite connection and the reranker belong to that thread.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .agent import AgentSettings, run_agent
from .profiling import format_table, reset, snapshot, timed
from .read import RULES, call_server
from .run import read_jsonl, select_questions, sha256
from .stages import PageIndex, run_pipeline

# The agent adds neighbours itself, so a shared search must not add them too.
SEARCH_STAGES = ("bm25", "rerank", "llm_select")


def search_steps(pipeline: list[dict]) -> list[dict]:
    return [step for step in pipeline if step["stage"] in SEARCH_STAGES]


def answer_one(
    question: dict,
    search: Callable,
    text_model: Callable[[str], str],
    vision_model: Callable[[list[dict]], str],
    data: Path,
    known: set,
    settings: AgentSettings,
) -> dict:
    """Run one question, scoring a failed request as empty so the run can continue."""
    row = {"question": question["question"], "id": question["id"]}
    try:
        result = run_agent(question["question"], search, text_model, vision_model, data, known, settings)
    except Exception as error:  # noqa: BLE001 - any failure, including GPU out-of-memory, fails one question only
        # Timeouts and dropped connections too: one failed question must never stop the run.
        print(f"REQUEST FAILED {question['id']}: {error}", flush=True)
        return {**row, "answer": [], "citations": [], "iterations": 1, "searches": [], "failed": True}
    return {
        **row,
        "answer": result.answer,
        "citations": result.citations,
        "iterations": max(result.iterations, 1),
        "searches": result.searches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rules", default="v3", choices=sorted(RULES))
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--pages-per-search", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=9)
    parser.add_argument("--visual", action="store_true", help="add ColQwen's best page to each search (query side on CPU)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-url", default="http://host.docker.internal:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    questions = select_questions(read_jsonl(args.data / "questions" / f"{args.split}.jsonl"), args.ids)[: args.limit]
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    index_path = args.data / "index" / f"{sha256(pages_path)[:16]}.sqlite"
    known = {(r["document"], r["page"]) for r in read_jsonl(args.data / "pages" / "pdf_text.jsonl")}
    steps = search_steps(config["pipeline"])

    reset()
    searcher = ThreadPoolExecutor(1)
    holder = {}

    def setup() -> None:
        pages = read_jsonl(pages_path) if not index_path.exists() else None
        holder["index"] = PageIndex(index_path, pages)
        holder["index"].pages_path = pages_path
        # Load page text once here, so parallel searches don't each load it.
        holder["index"]._load_text()
        first = next(iter(holder["index"]._text))
        for source in {step["text_from"] for step in steps if step.get("text_from")}:
            try:
                holder["index"].text(first, source)
            except KeyError:
                pass

    searcher.submit(setup).result()
    searcher.shutdown()

    colqwen = None
    if args.visual:
        from lab.visual import CpuSearcher, add_visual

        colqwen = CpuSearcher(args.data / "visual")
        print("ColQwen index loaded:", len(colqwen.keys), "pages", flush=True)

    def search(query: str) -> list:
        # Searches run in the worker threads. The reranker takes turns on the GPU behind a lock.
        with timed("agent_search"):
            ranked = run_pipeline(query, steps, holder["index"])
            if colqwen:
                ranked = add_visual(ranked, colqwen.search(query, 1), 1, 1, slots=args.pages_per_search)
            return ranked

    def text_model(prompt: str) -> str:
        reply = call_server(args.base_url, args.model, [{"type": "text", "text": prompt}])
        return reply["choices"][0]["message"]["content"] or ""

    def vision_model(content: list[dict]) -> str:
        reply = call_server(args.base_url, args.model, content)
        return reply["choices"][0]["message"]["content"] or ""

    settings = AgentSettings(
        max_rounds=args.max_rounds,
        pages_per_search=args.pages_per_search,
        max_images=args.max_images,
        rules=RULES[args.rules],
        max_side=args.max_side,
    )
    results = args.data / "results"
    label = (f"_{args.ids.stem}" if args.ids else "") + ("_colqwen" if args.visual else "")
    path = results / f"answers_agent_{args.model.replace('/', '_')}_{args.split}{label}.jsonl"
    # Each answer is saved as it finishes, so progress is visible and a restart resumes where it stopped.
    partial = path.with_suffix(".partial.jsonl")
    done = {row["id"]: row for row in read_jsonl(partial) if not row.get("failed")} if partial.exists() else {}
    todo = [q for q in questions if q["id"] not in done]
    print(f"Resuming: {len(done)} done, {len(todo)} to go", flush=True)
    write_lock = threading.Lock()
    finished = [len(done)]

    def run_and_save(question: dict) -> dict:
        row = answer_one(question, search, text_model, vision_model, args.data, known, settings)
        with write_lock, partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            finished[0] += 1
            print(f"progress {finished[0]}/{len(questions)}", flush=True)
        return row

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        fresh = {row["id"]: row for row in pool.map(run_and_save, todo)}
    seconds = time.perf_counter() - start
    outputs = [done.get(q["id"]) or fresh[q["id"]] for q in questions]
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in outputs)
    partial.unlink()
    entry = {
        "file": path.name,
        "questions_per_second": round(len(todo) / seconds, 3) if todo else None,
        "iterations": dict(sorted(Counter(o["iterations"] for o in outputs).items())),
        "planned_searches": dict(sorted(Counter(len(o["searches"]) for o in outputs).items())),
        "failed": sum(bool(o.get("failed")) for o in outputs),
        "profile": snapshot(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (results / "agent_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(path)
    print({k: entry[k] for k in ("questions_per_second", "iterations", "planned_searches", "failed")})
    print(format_table(entry["profile"]))


if __name__ == "__main__":
    main()
