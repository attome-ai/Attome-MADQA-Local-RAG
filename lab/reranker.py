"""Qwen-style yes/no cross-encoder, using the same prompt as SAG2's local reranker.

Needs torch and transformers, so it runs inside Docker. Scores are cached on disk, so repeated
experiments never recompute a question-page pair.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from .profiling import profiled, timed

INSTRUCTION = "Given a question, retrieve document pages containing the evidence that answers it."
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def cache_key(max_tokens: int, question: str, text: str) -> str:
    return hashlib.sha256(f"{max_tokens}\0{question}\0{text}".encode()).hexdigest()[:24]


class Reranker:
    def __init__(self, model: str, cache_path: Path, batch_size: int, max_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Batches of 4 fit the memory cap below (peak 1.7 GB) at the same speed as 8.
        self.torch, self.batch_size, self.max_tokens = torch, min(batch_size, 4), max_tokens
        # Hard cap: about 1.9 GB of the 32 GB card; with vLLM at 0.82 and the desktop, the total stays below the ~30.9 GiB where Windows starts spilling, so the reader never spills into system RAM.
        torch.cuda.set_per_process_memory_fraction(0.06)
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.engine = (
            AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa")
            .to("cuda")
            .eval()
        )
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        self.lock = threading.Lock()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    key, score = json.loads(line)
                    self.cache[key] = score

    def _encode(self, question: str, text: str) -> list[int]:
        head = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: "
        body = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_tokens]
        return self.prefix + self.tokenizer.encode(head, add_special_tokens=False) + body + self.suffix

    @profiled
    def score(self, question: str, texts: list[str]) -> list[float]:
        keys = [cache_key(self.max_tokens, question, text) for text in texts]
        # Searches run in parallel threads; one at a time uses the GPU and appends to the cache.
        with self.lock:
            self._fill(question, texts, keys)
            # It shares the GPU with the reader. Holding freed memory makes the card overflow into
            # system RAM over a long run, which stalls everything, so release it after each search.
            self.torch.cuda.empty_cache()
        return [self.cache[key] for key in keys]

    def _scores(self, question: str, texts: list[str]) -> list[float]:
        inputs = self.tokenizer.pad(
            {"input_ids": [self._encode(question, text) for text in texts]},
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with self.torch.inference_mode(), timed("reranker_batch"):
            logits = self.engine(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            return self.torch.softmax(logits[:, [self.no_id, self.yes_id]].float(), dim=-1)[:, 1].cpu().tolist()

    def _fill(self, question: str, texts: list[str], keys: list[str]) -> None:
        missing = sorted(
            (i for i, key in enumerate(keys) if key not in self.cache), key=lambda i: len(texts[i])
        )
        with self.cache_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                try:
                    scores = self._scores(question, [texts[i] for i in batch])
                except self.torch.OutOfMemoryError:
                    # A batch of long pages can exceed the memory cap; score them one at a time instead.
                    self.torch.cuda.empty_cache()
                    scores = [self._scores(question, [texts[i]])[0] for i in batch]
                for i, score in zip(batch, scores):
                    self.cache[keys[i]] = score
                    handle.write(json.dumps([keys[i], score]) + "\n")
