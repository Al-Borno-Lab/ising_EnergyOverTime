# E12 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 19:45 | finished | MacBook-Pro-140 | 42478c3 | - | `12_field_vs_rate_indicator/assets/figures/prelim_field_vs_rate_trough.png` | EXPLORATORY prelim figure (local): code/make_pi_figure.py on 09 field_term_trough_timing.csv; field pre 0.60 vs ctrl 0.14 p=2e-4; rate 0.32 vs 0.12 p=0.13; depth ratio 4.7x vs 2.0x p=0.12 |
| 2026-10-12 20:32 | submitted | MacBook-Pro-140 | e5d0027 (+uncommitted changes in experiment) | - | `12_field_vs_rate_indicator/results/synthetic` | METHOD VALIDATION (synthetic): validate_synthetic.py |
| 2026-10-12 20:35 | finished | MacBook-Pro-140 | e5d0027 (+uncommitted changes in experiment) | - | `-` | synthetic validation done -> assets/tables/synthetic_validation.csv |
