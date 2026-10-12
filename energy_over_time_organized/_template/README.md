# {{ID}} · {{TITLE}}

Status: see `experiment.json` · Owner: {{OWNER}} · Started: {{DATE}}

> This file is the **experiment write-up**: what we ask and how. Write it BEFORE running.
> Results go in `RESULTS.md`, never here.

## Question
What does this answer, and why does it matter for the paper?

## Hypothesis and decision rule
What result would support / refute it? If pre-registered, link the document and log changes as dated amendments.

## Data
Input files (cluster paths), sessions/conditions used, exclusions (`exclude == 1` reaches dropped?), bins (10 ms, rows 100:800).

## Method
Model, fitting procedure, statistics (unit = session, n = 21; Wilcoxon signed-rank; BH-FDR; cohort and mouse effects),
controls (rest/control windows for timing claims; shuffles/nulls).

## How to reproduce
```bash
# from the repo root, on the cluster (never run heavy work on the login node)
bash energy_over_time_organized/{{DIR}}/code/run.sh
```
List every script in `code/` and what it does:
- `code/run.sh`: entry point; submits/run everything in order.

## Outputs
- `results/` (cluster only, gitignored): raw outputs.
- `assets/figures/`, `assets/tables/` (committed): every figure/table cited in `RESULTS.md`.
