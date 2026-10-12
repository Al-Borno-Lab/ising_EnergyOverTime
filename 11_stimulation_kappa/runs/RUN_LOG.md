# E11 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 14:30 | submitted | MacBook-Pro-140 | 2617525 | - | `11_stimulation_kappa/results/validation` | synthetic validation (local, 13 procs): python code/validate_stimfit.py 13 results/validation 12 8; SYNTHETIC; 40 tasks; criteria VALIDATION_CRITERIA_2026-10-12.md |
| 2026-10-12 14:37 | failed | MacBook-Pro-140 | 2617525 (+uncommitted changes in experiment) | - | `11_stimulation_kappa/results/validation` | local validation aborted after 11/40 tasks (Mac shared with E10 run, too slow); rerun on cluster with identical code/criteria |
