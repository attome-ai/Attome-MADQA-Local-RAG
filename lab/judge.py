"""Score saved answers the way MADQA does: ANLS* first, then an LLM judge for anything imperfect.

    python -m lab.judge data/results/answers_x.jsonl [more files] --split dev

Uses MADQA's own prompt and rules. The model differs from MADQA's gemini-2.5-flash, and the
leaderboard's bias correction was calibrated for Gemini, so these are raw judged scores: good
for comparing our runs, and only an estimate of a leaderboard score.

The key comes from CMD_API_KEY or the project's .env. Verdicts are cached, so a repeated
question-answer pair is never paid for twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .madqa_metrics import _GEVAL_PROMPT_TEMPLATE, anls_star
from .metrics import hop_type
from .profiling import format_table, profiled, snapshot, timed
from .run import read_jsonl

BASE_URL = "https://api.commandcode.ai/provider/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
REPLY_FORMAT = (
    '\n\nInstead of calling a tool, end your reply with one line of JSON only: '
    '{"judgment": "correct" | "partial" | "incorrect", "main_issue": "...", "explanation": "..."}'
)
MAX_ANSWER_LENGTH = 2000


def load_key(env_path: Path = Path(".env")) -> str:
    key = os.environ.get("CMD_API_KEY")
    if not key and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMD_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise ValueError("CMD_API_KEY is not set")
    return key


def parse_judgment(text: str) -> str | None:
    """Return the verdict from the last JSON object holding one, or None if there is none."""
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("judgment") in SCORES:
            return data["judgment"]
    return None


def call_model(prompt: str, model: str, key: str, base_url: str = BASE_URL) -> str:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # The API's firewall rejects urllib's default User-Agent with error 1010.
            "User-Agent": "madqa-lab/0.1",
        },
    )
    with timed("judge_call"), urllib.request.urlopen(request, timeout=120) as response:
        reply = json.load(response)
    return reply["choices"][0]["message"]["content"] or ""


class Judge:
    def __init__(self, call: Callable[[str], str], cache_path: Path, retries: int = 4):
        self.call, self.cache_path, self.retries = call, cache_path, retries
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        self.errors = 0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            for key, verdict in read_jsonl(cache_path):
                self.cache[key] = verdict

    def verdict(self, question: str, predicted: list, gold: list) -> str | None:
        prompt = _GEVAL_PROMPT_TEMPLATE.format(
            question=question, predicted=json.dumps(predicted), gold_variants=json.dumps(gold)
        )
        key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        for attempt in range(self.retries):
            try:
                verdict = parse_judgment(self.call(prompt + REPLY_FORMAT))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                verdict = None
            if verdict:
                with self.lock:
                    self.cache[key] = verdict
                    with self.cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps([key, verdict]) + "\n")
                return verdict
            time.sleep(2**attempt)
        with self.lock:
            self.errors += 1
        return None


def judged_score(judge: Judge, question: dict, answer: list) -> tuple[float, float, bool]:
    """Return judged score, ANLS*, and whether the judge failed, following MADQA's rules."""
    gold = question["answer_variants"]
    if not answer or all(not item for item in answer):
        return 0.0, 0.0, False
    if len(json.dumps(answer)) > MAX_ANSWER_LENGTH:
        return 0.0, 0.0, False
    anls = anls_star(answer, gold)
    if anls >= 1.0:
        return 1.0, anls, False
    verdict = judge.verdict(question["question"], answer, gold)
    if verdict is None:
        return 0.0, anls, True
    return SCORES[verdict], anls, False


@profiled
def score_file(path: Path, questions: dict[str, dict], judge: Judge, workers: int = 6) -> dict:
    outputs = [o for o in read_jsonl(path) if o["id"] in questions]

    def one(output: dict) -> tuple[dict, float, float, bool]:
        question = questions[output["id"]]
        return (question, *judged_score(judge, question, output["answer"]))

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, outputs))

    def summary(items: list) -> dict:
        if not items:
            return {"n": 0}
        count = len(items)
        return {
            "n": count,
            "judged_accuracy": round(100 * sum(r[1] >= 0.5 for r in items) / count, 2),
            "judged_mean": round(100 * sum(r[1] for r in items) / count, 2),
            "text_match_accuracy": round(100 * sum(r[2] >= 0.5 for r in items) / count, 2),
            "judge_failures": sum(r[3] for r in items),
        }

    result = {"overall": summary(rows)}
    for hop in ("single", "cross_page", "cross_doc"):
        result[hop] = summary([r for r in rows if hop_type(r[0]) == hop])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    key = load_key()
    questions = {q["id"]: q for q in read_jsonl(args.data / "questions" / f"{args.split}.jsonl")}
    cache = args.data / "cache" / f"judge_{args.model.replace('/', '_')}.jsonl"
    judge = Judge(lambda prompt: call_model(prompt, args.model, key), cache)
    for path in args.files:
        metrics = score_file(path, questions, judge, args.workers)
        entry = {"file": path.name, "judge": args.model, "split": args.split, "metrics": metrics}
        with (args.data / "results" / "judged_ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        overall = metrics["overall"]
        hops = [metrics[h].get("judged_accuracy") for h in ("single", "cross_page", "cross_doc")]
        print(f"{path.name}\n  {overall}\n  by type (single, cross-page, cross-doc): {hops}", flush=True)
    print(format_table(snapshot()))


if __name__ == "__main__":
    main()
