"""Read, verify and workspace back to back per question, so the reader's cache still holds the pages.

    python -m lab.chain experiments/thin_ocr_llm_select.json --model NAME --split dev [--ids set.json]

Same prompts and pages as running lab.answer (v3), lab.verify and lab.answer (workspace) separately; only
the order changes. Run separately, a question's pages leave the cache before the next pass reaches them
(about 4 requests fit in memory), so every pass re-reads them from scratch.
Writes the same three files as the separate runs, with "_chain" added to each name.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import answer_questions, saved_rankings
from .profiling import format_table, reset, snapshot
from .read import RULES, call_server
from .run import read_jsonl, select_questions, sha256
from .verify import verify_questions


def chain_one(question: dict, ranked: dict, call, data: Path, top_k: int, max_side: int) -> tuple[dict, dict, dict]:
    retrieve = lambda text: ranked[question["id"]]
    draft = answer_questions([question], retrieve, call, data, top_k, 1, RULES["v3"], max_side, True)[0]
    verified = verify_questions([draft], ranked, call, data, top_k, RULES["v3"], max_side, 1)[0]
    workspace = answer_questions([question], retrieve, call, data, top_k, 1, RULES["workspace"], max_side, True)[0]
    return draft, verified, workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pages_path = args.data / "pages" / f"{config['pages']}.jsonl"
    ranked = saved_rankings(args.data, config["name"], args.split, sha256(pages_path))
    questions = select_questions(read_jsonl(args.data / "questions" / f"{args.split}.jsonl"), args.ids)
    call = lambda content: call_server(args.base_url, args.model, content)
    reset()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda q: chain_one(q, ranked, call, args.data, args.top_k, args.max_side), questions))
    seconds = time.perf_counter() - start
    stem = f"answers_{config['name']}_{args.model.replace('/', '_')}_k{args.top_k}_{args.split}"
    ids = f"_{args.ids.stem}" if args.ids else ""
    names = [f"v3_px{args.max_side}_labels", f"v3_px{args.max_side}_labels_verified", f"workspace_px{args.max_side}_labels"]
    for position, name in enumerate(names):
        path = args.data / "results" / f"{stem}_{name}{ids}_chain.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r[position], ensure_ascii=False) + "\n" for r in rows)
        print(path)
    print(f"Questions per second (all three methods): {len(questions) / seconds:.3f}")
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
