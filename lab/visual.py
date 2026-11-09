"""Visual page retrieval with ColQwen2.5 (T18). Runs inside the madqa-colpali image (GPU, no vLLM alongside).

    python -m lab.visual embed                 # embed every page of every PDF, saved per document
    python -m lab.visual eval --split dev      # recall of ColQwen alone, and added to the saved ranking

Each page is rendered and embedded as a bag of 128-dimension vectors. A query scores a page by MaxSim: for each
query vector the best matching page vector, summed. The point is to find pages whose text layer or OCR lacks the
words (images, charts, merged OCR words) that BM25 cannot match.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = "vidore/colqwen2.5-v0.2"
MAX_PIXELS = 640 * 28 * 28  # about 640 visual tokens per page
RENDER_SIDE = 1100


def load_model():
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL)
    processor.image_processor.max_pixels = MAX_PIXELS
    return model, processor


def embed(data: Path, out: Path, batch: int = 8) -> None:
    import pypdfium2
    import torch

    out.mkdir(parents=True, exist_ok=True)
    model, processor = load_model()
    pdfs = sorted((data / "pdfs").glob("*.pdf"))
    started = time.time()
    done_pages = 0
    for number, path in enumerate(pdfs, 1):
        target = out / f"{path.stem}.pt"
        if target.exists():
            continue
        pdf = pypdfium2.PdfDocument(path)
        vectors = []
        try:
            images = []
            for index in range(len(pdf)):
                page = pdf[index]
                scale = RENDER_SIDE / max(page.get_size())
                images.append(page.render(scale=scale).to_pil().convert("RGB"))
        finally:
            pdf.close()
        for start in range(0, len(images), batch):
            inputs = processor.process_images(images[start : start + batch]).to(model.device)
            with torch.no_grad():
                embeddings = model(**inputs)
            for row, mask in zip(embeddings, inputs["attention_mask"]):
                vectors.append(row[mask.bool()].cpu().to(torch.float16))
        torch.save(vectors, target)
        done_pages += len(vectors)
        if number % 20 == 0:
            rate = done_pages / (time.time() - started)
            print(f"{number}/{len(pdfs)} documents, {done_pages} pages, {rate:.1f} pages/s", flush=True)
    print("embedded", len(pdfs), "documents", flush=True)


def load_index(folder: Path):
    import torch

    keys, tensors = [], []
    for path in sorted(folder.glob("*.pt")):
        for page, vectors in enumerate(torch.load(path), 1):
            keys.append((f"{path.stem}.pdf", page))
            tensors.append(vectors)
    return keys, tensors


def search_all(model, processor, queries: list[str], keys, tensors, top: int = 50, chunk: int = 256):
    import torch

    results = []
    for number, query in enumerate(queries, 1):
        if number % 100 == 0:
            print(f"queried {number}/{len(queries)}", flush=True)
        inputs = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            q = model(**inputs)[0].to(torch.bfloat16)
        scores = []
        for start in range(0, len(tensors), chunk):
            block = tensors[start : start + chunk]
            padded = torch.nn.utils.rnn.pad_sequence(block, batch_first=True).to(q.device, torch.bfloat16)
            sims = torch.einsum("qd,npd->nqp", q, padded)
            scores.append(sims.max(dim=2).values.sum(dim=1).float().cpu())
        merged = torch.cat(scores)
        best = merged.topk(min(top, len(keys))).indices.tolist()
        results.append([keys[i] for i in best])
    return results


class CpuSearcher:
    """ColQwen query side on the CPU, for the agent: the GPU is full of the reader, so only queries are encoded here.

    The page vectors stay in RAM (about 3.4 GB). One query at a time, behind a lock.
    """

    def __init__(self, folder: Path):
        import threading

        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu").eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(MODEL)
        self.keys, self.tensors = load_index(folder)
        self.lock = threading.Lock()

    def search(self, query: str, top: int = 5) -> list:
        import torch

        with self.lock, torch.no_grad():
            inputs = self.processor.process_queries([query])
            q = self.model(**inputs)[0].to(torch.float32)
            scores = []
            for start in range(0, len(self.tensors), 128):
                block = torch.nn.utils.rnn.pad_sequence(self.tensors[start : start + 128], batch_first=True).to(torch.float32)
                scores.append(torch.einsum("qd,npd->nqp", q, block).max(dim=2).values.sum(dim=1))
            best = torch.cat(scores).topk(top).indices.tolist()
        return [self.keys[i] for i in best]


def latest_ranking(data: Path, name: str, split: str) -> dict[str, list]:
    ledger = [json.loads(line) for line in (data / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    matches = [e for e in ledger if e["name"] == name and e["split"] == split and e["limit"] is None]
    if not matches:
        return {}
    path = data / "results" / "runs" / f"{matches[-1]['key']}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["id"]: [tuple(p) for p in row["ranked"]] for row in rows}


def evaluate(data: Path, folder: Path, split: str, baseline: str) -> None:
    model, processor = load_model()
    keys, tensors = load_index(folder)
    print("pages indexed", len(keys), flush=True)
    questions = [json.loads(l) for l in (data / "questions" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    base = latest_ranking(data, baseline, split)
    ranked = search_all(model, processor, [q["question"] for q in questions], keys, tensors)
    out = data / "results" / f"visual_{split}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for question, pages in zip(questions, ranked):
            handle.write(json.dumps({"id": question["id"], "ranked": pages}) + "\n")
    gold = lambda q: {(e["document"], int(e["page"])) for e in q["evidence"]}  # noqa: E731
    print(f"{split}: {len(questions)} questions; baseline '{baseline}' rankings: {len(base)}")
    for k in (3, 5, 9, 20, 50):
        vis = sum(len(gold(q) & set(r[:k])) / len(gold(q)) for q, r in zip(questions, ranked)) / len(questions)
        line = f"  ColQwen alone recall@{k}: {100 * vis:.2f}"
        if base:
            both = sum(
                len(gold(q) & (set(base.get(q["id"], [])[:9]) | set(r[: k]))) / len(gold(q))
                for q, r in zip(questions, ranked)
            ) / len(questions)
            only = sum(len(gold(q) & set(base.get(q["id"], [])[:9])) / len(gold(q)) for q in questions) / len(questions)
            line += f"   baseline top 9: {100 * only:.2f}; baseline top 9 + ColQwen top {k}: {100 * both:.2f}"
        print(line)


def add_visual(base: list, visual: list, take: int, depth: int, slots: int = 9) -> list:
    """Swap the last slots of the reader's pages for ColQwen's best pages that the text ranking missed."""
    extras = [p for p in visual[:depth] if p not in base[:slots]][:take]
    if not extras:
        return list(base)
    kept = base[: slots - len(extras)]
    return list(dict.fromkeys(kept + extras + base[slots - len(extras) :]))


def merge(data: Path, split: str, baseline: str, name: str, take: int, depth: int) -> None:
    """Save a ranking in the ledger, so lab.answer and lab.incremental can use it like any experiment."""
    import hashlib

    ledger = [json.loads(line) for line in (data / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    base_entry = [e for e in ledger if e["name"] == baseline and e["split"] == split and e["limit"] is None][-1]
    base = latest_ranking(data, baseline, split)
    visual = {
        row["id"]: [tuple(p) for p in row["ranked"]]
        for row in map(json.loads, (data / "results" / f"visual_{split}.jsonl").read_text(encoding="utf-8").splitlines())
    }
    key = hashlib.sha256(f"{name}-{split}-{take}-{depth}".encode()).hexdigest()[:16]
    changed = 0
    with (data / "results" / "runs" / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
        for question_id, ranked in base.items():
            merged = add_visual(ranked, visual[question_id], take, depth)
            changed += merged[:9] != ranked[:9]
            handle.write(json.dumps({"id": question_id, "ranked": merged}) + "\n")
    entry = {**base_entry, "key": key, "name": name, "metrics": {}, "profile": {}}
    entry["hypothesis"] = f"ColQwen pages the text ranking missed take the last {take} of 9 reader slots (top {depth} of ColQwen)."
    with (data / "results" / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"{name} on {split}: top 9 changed for {changed} of {len(base)} questions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["embed", "eval", "merge"])
    parser.add_argument("--name", default="thin_ocr_visual")
    parser.add_argument("--take", type=int, default=2)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--baseline", default="thin_ocr_llm_select")
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    folder = args.data / "visual"
    if args.mode == "embed":
        embed(args.data, folder)
    elif args.mode == "merge":
        merge(args.data, args.split, args.baseline, args.name, args.take, args.depth)
    else:
        evaluate(args.data, folder, args.split, args.baseline)


if __name__ == "__main__":
    main()
