# E11 run log

Append with `bash tools/log_run.sh <this folder> <submitted|finished|failed> "<text>" [job_id] [output_dir]`.
Never edit past rows; failed runs stay in the log.

| When | Event | Host | Commit | SLURM job | Output | Notes |
|---|---|---|---|---|---|---|
| 2026-10-12 14:30 | submitted | MacBook-Pro-140 | 2617525 | - | `11_stimulation_kappa/results/validation` | synthetic validation (local, 13 procs): python code/validate_stimfit.py 13 results/validation 12 8; SYNTHETIC; 40 tasks; criteria VALIDATION_CRITERIA_2026-10-12.md |
| 2026-10-12 14:37 | failed | MacBook-Pro-140 | 2617525 (+uncommitted changes in experiment) | - | `11_stimulation_kappa/results/validation` | local validation aborted after 11/40 tasks (Mac shared with E10 run, too slow); rerun on cluster with identical code/criteria |
| 2026-10-12 14:37 | submitted | MacBook-Pro-140 | 0ac4bc3 | 4362873 | `~/ising_wt/e11/results/validation` | sbatch code/sbatch_validate.sh (40 procs): validate_stimfit.py 40 results/validation 12 8; SYNTHETIC; 40 tasks; same criteria |
| 2026-10-12 14:45 | finished | MacBook-Pro-140 | 782b3b2 | 4362873 | `-` | validation run 1 (job 4362873): FAIL criterion 2 in scenario C (dkappa 0.43); A,B,D,E and crit 3-5 pass; see VALIDATION_CRITERIA run-1 section |
| 2026-10-12 14:45 | submitted | MacBook-Pro-140 | 782b3b2 (+uncommitted changes in experiment) | 4362884 | `~/ising_wt/e11/results/validation_run2` | validation run 2 after amendment 2 (lamD 1e-5); sbatch code/sbatch_validate.sh results/validation_run2; SYNTHETIC |
| 2026-10-12 14:53 | finished | MacBook-Pro-140 | 0727e13 (+uncommitted changes in experiment) | 4362884 | `-` | validation run 2 (job 4362884, lamD 1e-5): PASS all 5 criteria (A 0.451, E 0.476, B 0.080, C 0.025, D 0.021) |
| 2026-10-12 15:36 | submitted | MacBook-Pro-140 | f4d7d3b | 4362993 | `~/ising_wt/e11/results/validation_run3` | validation run 3 (amendment 3: pseudo-onset control, fit bins 300-559, R=60 x8 and R=18 x16 per scenario); sbatch code/sbatch_validate.sh results/validation_run3; SYNTHETIC; 120 tasks |
| 2026-10-12 15:44 | finished | MacBook-Pro-140 | 4587ac7 | 4362993 | `-` | validation run 3 (job 4362993): FAIL crit 4 (117/120 hit iteration cap, cropped-basis edge columns) and R18 scenario C runaway fit; see amendment 4 |
| 2026-10-12 15:44 | submitted | MacBook-Pro-140 | 4587ac7 (+uncommitted changes in experiment) | 4363017 | `~/ising_wt/e11/results/validation_run4` | validation run 4 (amendment 4: cropped basis); sbatch code/sbatch_validate.sh results/validation_run4; SYNTHETIC; 120 tasks |
| 2026-10-12 15:48 | finished | MacBook-Pro-140 | 9aeb2dd | 4363017 | `-` | validation run 4 (job 4363017): PASS all criteria, R=60 and R=18 |
| 2026-10-12 15:50 | submitted | MacBook-Pro-140 | 877dbaa | 4363032 | `~/ising_wt/e11_repo/11_stimulation_kappa/results/smoke` | SMOKE real data, session 210511 (N=12): primary,round,shift1,fixedctrl,allreach,s3 + 2 shuffles; code commit 877dbaa (clone ~/ising_wt/e11_repo); prereg 0e23c6d |
| 2026-10-12 15:50 | submitted | MacBook-Pro-140 | 877dbaa (+uncommitted changes in experiment) | 4363033 | `~/ising_wt/e11_repo/11_stimulation_kappa/results/smoke17` | SMOKE/cost timing real data, session 220519 (N=17) subset 0 primary only; code commit 877dbaa |
| 2026-10-12 15:55 | submitted | MacBook-Pro-140 | efba9d9 | 4363045 | `~/ising_wt/e11_repo/11_stimulation_kappa/results/diag_*` | DIAGNOSIS real data 210511 subset 0 primary + shuf_0 under 4 fit settings (base, tol 1e-7, lamB 1e-4, both) to understand slow convergence / runaway shuffle; not an analysis run; code commit efba9d9 |
| 2026-10-12 16:01 | submitted | MacBook-Pro-140 | a0a3177 (+uncommitted changes in experiment) | 4363048 | `~/ising_wt/e11_repo/11_stimulation_kappa/results/validation_run5` | validation run 5 (amendment 5: tol 1e-7); sbatch from clone ~/ising_wt/e11_repo; SYNTHETIC; 120 tasks; code commit a0a3177 |
| 2026-10-12 16:03 | finished | MacBook-Pro-140 | 01836b9 (+uncommitted changes in experiment) | 4363048 | `-` | validation run 5 (job 4363048, tol 1e-7): PASS all criteria R=60 and R=18 |
| 2026-10-12 16:03 | failed | MacBook-Pro-140 | 01836b9 (+uncommitted changes in experiment) | 4363033 | `-` | smoke17 (220519 subset 0, tol 1e-9) cancelled after 12+ min to retime with the final tol 1e-7 (amendment 5) |
| 2026-10-12 16:03 | submitted | MacBook-Pro-140 | 909c083 | 4363049 | `~/ising_wt/e11_repo/11_stimulation_kappa/results/time17` | COST TIMING real data session 220519 (N=17) subset 0 primary only, final tol 1e-7; code commit 01836b9 |
| 2026-10-12 16:15 | finished | MacBook-Pro-140 | 11b33a1 | 4363049 | `-` | cost timing 220519 subset 0 primary (N=17 -> 16 used): 686 s, 306 its, converged; smoke 210511 (N=12) tasks finished: see prereg A2 for disclosure; smoke results NOT used as results |
