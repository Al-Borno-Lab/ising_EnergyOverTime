# E14 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-13 02:13 | submitted | MacBook-Pro-140 | 68c6172 | 4363528 | `~/ising_wt/e14/14_binsize_1ms_synchrony/results/spk` | extraction + data gates G1-G3 for all 21 sessions with the integer-sample binning rule (after diagnose_g1/diagnose_ties on 210512, 210619, 220516); copies raw files under HOME and deletes them; 8 CPUs; code commit 68c6172 |
| 2026-10-13 03:52 | finished | MacBook-Pro-140 | 1c867bd | 4363528 | `-` | extraction + gates rerun job 4363528 COMPLETED (1m36s); gate results not yet read (paused) |
