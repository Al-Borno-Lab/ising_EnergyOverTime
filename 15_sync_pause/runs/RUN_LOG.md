# E13 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588232 | `15_sync_pause/results` | validate (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588233 | `15_sync_pause/results` | synth (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588234 | `15_sync_pause/results` | real (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
| 2026-10-27 21:25 | failed | MacBook-Pro-140 | 21cf2e6 | 4588234 | `-` | real tasks 5 (210512) and 9 (210608) stopped on fold assertion (a condition lacks a fold after exclusions); fixed by amendment A1 |
| 2026-10-27 21:25 | submitted | MacBook-Pro-140 | 21cf2e6 | 4588294 | `15_sync_pause/results` | real --array=5,9 rerun after amendment A1 (commit 21cf2e6) |
| 2026-10-27 21:25 | finished | MacBook-Pro-140 | 21cf2e6 | 4588232 | `-` | validate_lib: exact=bruteforce (2e-16); Gibbs vs exact max\|dP\| <= 0.0053, pause prob within 5e-4 |
