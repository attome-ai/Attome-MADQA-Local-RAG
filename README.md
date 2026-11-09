# Attome MADQA Local RAG

A fully local agentic RAG system for **MADQA** (`OxRML/MADQA`): questions over 799 multi-page PDFs, where the
answer often sits in a table, a form or a chart and sometimes needs several pages or documents. Every model that
reads, selects, checks or answers runs locally (Qwen3.8-27B on vLLM, ColQwen, Qwen3-Reranker). The only remote
model is the **judge** used to score answers during development (`deepseek/deepseek-v4.1-flash`).

## Results

Answers are scored by the judge; a question counts as right at a judged score of 0.5 or more.

| Set | Questions | Right | Accuracy | Note |
|---|---|---|---|---|
| Dev | 200 | 189 | **94.5%** | Tuned on, so optimistic (noise is about ±5) |
| Train, 300 untuned | 300 | 285 | **95.0%** | Held out from all tuning |
| Train, mix of 200 | 200 | 185 | **92.5%** | Untuned, test-like mix |
| Train, 50 list questions | 50 | 47 | **94.0%** | Untuned, the hardest type |
| Test | 500 | n/a | not run | Never run, nothing submitted to the leaderboard |

Expected test accuracy is about **93%**. For reference, the public leaderboard has 94.4 as the best open-weight
system and 96.9 overall.

## What moved the score

| Change | Measured on | Before | After |
|---|---|---|---|
| OCR for near-empty pages (under 20 characters) | Dev page recall@5 | 47.1 | 63.5 |
| Qwen3-Reranker-0.6B over BM25 top 100 | Dev page recall@5 | 63.5 | 71.0 |
| Recovering 123 missing PDFs (799 of 800) | End to end, text only | 51.5 | 61.0 |
| ColQwen visual retrieval swapped into the last reader slots | Dev right, of 200 | 187 | 189 |
| Same, on untuned train (142 changed questions) | Right | 87 | 114 (+30 / -3) |
| Adjudicator over verify, workspace and agent answers | Dev right, of 200 (early version) | 87.5% | 90.5% (+6 / -0) |

## Pipeline

| Step | What it does |
|---|---|
| 1. Retrieve | BM25 over OCR text, then Qwen3-Reranker top 100, then LLM page selection |
| 2. Add visual pages | ColQwen finds pages the text ranking missed |
| 3. Read | Qwen3.8-27B reads the top 3 pages plus neighbours (up to 9 images at 2,048 px) |
| 4. Candidates | Verify pass, workspace read and an agent that searches on its own |
| 5. Adjudicate | One call picks among the candidates only when they disagree |
| 6. Units | Fixes units and formatting of the final answer |

## What did not work

Each idea was kept only if it held up on untuned train questions, not only on dev. These did not:

| Idea | Result |
|---|---|
| Deep read of every page of a document (extract, merge, clean) | Recovers missing list items, adds as many wrong ones |
| Two-pass agreement, stricter per-item checks, quote grounding | Pooled +1 / -5 for grounding; none separated real items from wrong ones |
| Slot filling (look up named parts the answer lacks) | +0 / -1; only about 2% of questions name countable parts |
| Page notes (25 extra pages turned into text notes) | 29 to 24 right on 37 multi-page dev questions (+1 / -6) |
| Few-shot examples, thinking mode, document regrouping, hybrid image + OCR read | No gain or worse |
| Larger or API models as the reader | Ruled out on purpose: the system is local only |

## Where the remaining errors come from

| Cause | Detail |
|---|---|
| Gold page never shown to the reader | Multi-page questions: 72% of train have every gold page in the reader's 9 pages (single-page: 97.6%) |
| Lists that span many pages | The reader cannot tell a complete list from a partial one |
| Ambiguous or noisy gold labels | A few dev questions |

## Learned page ranker

A gradient-boosted ranker (`lab/pagerank.py`) chooses the reader's 9 pages from the text rank, ColQwen rank and
document context, trained on the gold pages of train with out-of-fold scores.

| Measure | Current top 9 | Ranker top 9 |
|---|---|---|
| Train multi-page, all gold pages shown (5-fold) | 171 / 237 | 195 / 237 |
| Train single-page, all gold pages shown (5-fold) | 1282 / 1313 | 1297 / 1313 |
| Plain read on 207 untuned train questions | 168 right | 185 right (+24 / -7) |

Full pipeline (read, verify, workspace, adjudicate, units), only on questions where the ranker changes which gold pages are shown: dev 3 to 4 right of 9 (+1 / -0); untuned train 34 to 41 right of 52 (+11 / -4). Worth about +0.5 point overall, so it is kept. It is not yet exported for the test split.

## Layout

| Path | Contents |
|---|---|
| `lab/` | Pipeline code: retrieval stages, reader, verify, adjudicate, units, agent, experiments |
| `experiments/` | Experiment configs and question id sets |
| `EXPERIMENT_LOG.md` | Every experiment with its result, kept or dropped |
| `docker/` | Dockerfiles for OCR and ColQwen |

`data/` (PDFs, indexes, caches, results) and `.env` are not in the repository.

---

## Experiment lab

Try many retrieval and agent ideas cheaply, and let measured results decide what stays.

## Setup

1. Accept the terms at https://huggingface.co/datasets/OxRML/MADQA
2. Run:

```bash
uv sync
uv run hf auth login
uv run python -m lab.prepare --limit 10   # smoke test, then rerun without --limit
```

Check the page numbering line that `prepare` prints. Offset 0 should score highest. If another
offset wins, fix the numbering before trusting any recall figures.

## Run an experiment

```bash
uv run python -m lab.run experiments/bm25_baseline.json --split dev
uv run python -m lab.report --split dev --k 10
```

Each run is recorded in `data/results/ledger.jsonl`. Identical runs are skipped.

## Add an idea

1. Add a function decorated with `@stage` in `lab/stages.py`.
2. Copy a config in `experiments/` and write down the hypothesis.
3. Run it and compare in the report.

## Rules

- Tune on train and dev only. `lab.run` refuses the test split.
- Compare ideas on the full split. Runs with `--limit` stay out of the report.
- Small gaps on dev (200 questions) can be noise, so confirm promising ideas on train.
