# E10 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 14:11 | submitted | MacBook-Pro-140 | d680b07 | - | `10_operating_point_h2_h4/results/step1` | step1 synthetic (local, 12 procs): python code/step1_synthetic_amendments.py 12 results/step1 results/step1/h0_truth_J.npz; SYNTHETIC; 80 tasks |
| 2026-10-12 14:23 | failed | MacBook-Pro-140 | d680b07 (+uncommitted changes in experiment) | - | `-` | step1 first launch crashed after first task: relative OUT path broken by os.chdir (bug, fixed: OUT made absolute); no results kept |
| 2026-10-12 14:23 | submitted | MacBook-Pro-140 | 5ab6dca | - | `10_operating_point_h2_h4/results/step1` | step1 synthetic relaunch (local, 12 procs), same command, SYNTHETIC; 80 tasks |
| 2026-10-12 16:09 | finished | MacBook-Pro-140 | 5ab6dca (+uncommitted changes in experiment) | - | `-` | step1 synthetic done (80/80 tasks, ~1 h). kappa test PASS (A, C); rho_real-rho_null PASS in D, FAIL in E (null reproduces effect). Gate FAIL -> step 2 held. See assets/tables/step1_summary.csv |
| 2026-10-12 16:13 | submitted | MacBook-Pro-140 | 8c667c9 | 4363051,4363052 | `~/ising_wt/e10/10_operating_point_h2_h4/results/smoke*` | step2 SMOKE (cluster ~/ising_wt/e10): 220516_2 (N=19, nshuf=1) job 4363051; 210608_2 (N=21->1 subset of 16, nshuf=1) job 4363052; timing only, not for analysis |
| 2026-10-12 16:14 | failed | MacBook-Pro-140 | 2391744 (+uncommitted changes in experiment) | 4363051,4363052 | `-` | step2 smoke jobs 4363051/4363052 failed in 2 s: sbatch spool copy broke dirname $0 (script bug, fixed: cd to SLURM_SUBMIT_DIR) |
| 2026-10-12 16:14 | submitted | MacBook-Pro-140 | 88b1ffe | 4363053,4363054 | `~/ising_wt/e10/10_operating_point_h2_h4/results/smoke*` | step2 SMOKE resubmit after sbatch cwd fix: 220516_2 (N=19, nshuf=1) job 4363053; 210608_2 (N=21->1 subset, nshuf=1) job 4363054; timing only |
| 2026-10-12 16:15 | failed | MacBook-Pro-140 | f50f4d2 (+uncommitted changes in experiment) | 4363053,4363054 | `-` | step2 smoke jobs 4363053/4363054 failed: container entrypoint changes cwd to HOME, relative script path not found (fixed: absolute path) |
| 2026-10-12 16:15 | submitted | MacBook-Pro-140 | 3a16795 | 4363055,4363056 | `~/ising_wt/e10/10_operating_point_h2_h4/results/smoke*` | step2 SMOKE resubmit 3 (absolute path): 220516_2 (N=19, nshuf=1) job 4363055; 210608_2 (N=21->1 subset, nshuf=1) job 4363056; timing only |
| 2026-10-12 16:16 | failed | MacBook-Pro-140 | 34e5494 (+uncommitted changes in experiment) | 4363055,4363056 | `-` | step2 smoke jobs 4363055/4363056 failed: container cannot see /home/ceas/... real path of SLURM_SUBMIT_DIR (fixed: $HOME-based paths) |
| 2026-10-12 16:17 | submitted | MacBook-Pro-140 | 354d556 | 4363058,4363059 | `~/ising_wt/e10/10_operating_point_h2_h4/results/smoke*` | step2 SMOKE resubmit 4 (HOME paths): 220516_2 (N=19, nshuf=1) job 4363058; 210608_2 (N=21->1 subset, nshuf=1) job 4363059; timing only |
| 2026-10-12 17:09 | finished | MacBook-Pro-140 | 7505446 | 4363059 | `-` | step2 smoke 210608_2 (N=16 subset, nshuf=1): ok, 3089 s, 1.7 GB; N=19 smoke 4363058 still running after 52 min |
| 2026-10-12 17:09 | submitted | MacBook-Pro-140 | 78aa94a | 4363115 | `~/ising_wt/e10/10_operating_point_h2_h4/results/step2` | step2 FULL (pre-registered settings: 10 subsets for N>20, 20 shuffles, 63 conditions) at code commit 78aa94a; 64 cpus, 230G, 36h; per-task pkl, resumable |
