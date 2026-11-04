# E13 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-11-03 20:12 | submitted | MacBook-Pro-140 | 9f5d089 | 5058414 | `16_onems_timing/results/synth` | synthetic Z0/Z1, 6 conditions (code/run.sh submit) |
| 2026-11-03 20:12 | submitted | MacBook-Pro-140 | 9f5d089 | 5058415 | `16_onems_timing/results/real` | real: 63 conditions x 3 bases, one task per session |
| 2026-11-03 20:17 | finished | MacBook-Pro-140 | ab08f8e | 5058414 | `-` | synthetic: Z1 PASS (5/6 peak, 5/6 trough hits); Z0 FAIL for peak (D_peak -0.33, conservative direction: reach-window argmax pushed to window edge by broad dip), trough OK (+0.08). Real results NOT read. |
| 2026-11-03 20:17 | finished | MacBook-Pro-140 | ab08f8e | 5058415 | `-` | real: 21/21 sessions completed; pulled, NOT read pending decision on Z0 |
