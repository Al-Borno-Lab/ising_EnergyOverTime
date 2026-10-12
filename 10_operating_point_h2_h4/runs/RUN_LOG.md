# E10 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 14:11 | submitted | MacBook-Pro-140 | d680b07 | - | `10_operating_point_h2_h4/results/step1` | step1 synthetic (local, 12 procs): python code/step1_synthetic_amendments.py 12 results/step1 results/step1/h0_truth_J.npz; SYNTHETIC; 80 tasks |
| 2026-10-12 14:23 | failed | MacBook-Pro-140 | d680b07 (+uncommitted changes in experiment) | - | `-` | step1 first launch crashed after first task: relative OUT path broken by os.chdir (bug, fixed: OUT made absolute); no results kept |
