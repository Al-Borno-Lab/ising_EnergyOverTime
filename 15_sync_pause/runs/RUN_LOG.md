# E13 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588232 | `15_sync_pause/results` | validate (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588233 | `15_sync_pause/results` | synth (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
| 2026-10-27 21:24 | submitted | MacBook-Pro-140 | f052615 | 4588234 | `15_sync_pause/results` | real (sbatch via code/run.sh submit, cluster worktree ~/ising_wt/e15) |
