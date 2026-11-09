# MADQA experiment lab

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
