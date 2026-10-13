# E12 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 19:45 | finished | MacBook-Pro-140 | 42478c3 | - | `12_field_vs_rate_indicator/assets/figures/prelim_field_vs_rate_trough.png` | EXPLORATORY prelim figure (local): code/make_pi_figure.py on 09 field_term_trough_timing.csv; field pre 0.60 vs ctrl 0.14 p=2e-4; rate 0.32 vs 0.12 p=0.13; depth ratio 4.7x vs 2.0x p=0.12 |
| 2026-10-12 20:32 | submitted | MacBook-Pro-140 | e5d0027 (+uncommitted changes in experiment) | - | `12_field_vs_rate_indicator/results/synthetic` | METHOD VALIDATION (synthetic): validate_synthetic.py |
| 2026-10-12 20:35 | finished | MacBook-Pro-140 | e5d0027 (+uncommitted changes in experiment) | - | `-` | synthetic validation done -> assets/tables/synthetic_validation.csv |
| 2026-10-12 20:36 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `12_field_vs_rate_indicator/results/primary` | REAL primary: compute_signals.py (exclude_k==0) + stats_e12.py |
| 2026-10-12 20:38 | finished | MacBook-Pro-140 | 3bd7f2a | - | `-` | real primary done |
| 2026-10-12 20:38 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `12_field_vs_rate_indicator/results/allreaches` | REAL sensitivity allreaches (--keep-all) |
| 2026-10-12 20:39 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `~/ising_wt/e12/raw` | copy 21 raw NPresults files (~14 GB) to ~/ising_wt/e12/raw for kinematics extraction (login-node cp) |
| 2026-10-12 20:40 | finished | MacBook-Pro-140 | 3bd7f2a | - | `-` | sensitivity allreaches done |
| 2026-10-12 20:40 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `12_field_vs_rate_indicator/results/minN5` | REAL sensitivity minN5 (--min-n 5) |
| 2026-10-12 20:42 | finished | MacBook-Pro-140 | 3bd7f2a | - | `-` | sensitivity minN5 done |
| 2026-10-12 20:42 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `12_field_vs_rate_indicator/results/sigma2` | REAL sensitivity sigma2 (--sigma 2) |
| 2026-10-12 20:44 | finished | MacBook-Pro-140 | 3bd7f2a | - | `-` | sensitivity sigma2 done |
| 2026-10-12 20:44 | submitted | MacBook-Pro-140 | 3bd7f2a | - | `12_field_vs_rate_indicator/results/sigma5` | REAL sensitivity sigma5 (--sigma 5) |
| 2026-10-12 20:46 | finished | MacBook-Pro-140 | 3bd7f2a | - | `-` | sensitivity sigma5 done |
| 2026-10-12 20:47 | failed | MacBook-Pro-140 | 3bd7f2a | - | `~/ising_wt/e12` | cluster kinematics step: sbatch of extract_kinematics.py was DENIED by the session's permission classifier (shared-cluster mutation); raw copy (~16 GB) already sits in ~/ising_wt/e12/raw and is still to be deleted; POOL and the rest check pending the user's decision |
