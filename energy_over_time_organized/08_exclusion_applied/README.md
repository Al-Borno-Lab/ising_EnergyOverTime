# 08_exclusion_applied: MCH refit with Dylan's excluded reaches removed (started 2026-10-07)

**Why.** Dylan's source files (`/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_*_NPresults_short.mat`) flag 157 of the 1249 analysed reaches as `ReachS.exclude = 1`. The original `*_bin10_spike.mat` files predate the exclude export, so every flagged reach was used in `01_ising_model_fitting/results`. Decision: drop them, refit, and report both versions.

**Inputs.** `/data001/projects/enserrog/AbigailData/energy_over_time_exclude_applied/*_bin10_spike.mat`: copies of the originals with flagged reaches removed (1249 → 1092 reaches; the smallest block keeps 8). Built by `code/make_excluded_copies.py` and reloaded with `utils.loadmat` as a check. Per-block counts are in `code/exclusion_filter_log.csv`.

**Fitting.** `code/run_mch_fits_no_arbitration.sh` is byte-identical to `energy_over_time_script/process_matlab_file_manyTimes_sbatch.sh` up to `rm job_template.sh`. The arbitration follow-up is removed. Same `main.py` arguments (MCH, max_iter 200, eta 0.05, sample_size 1e5, rows 100:800, full_reach, 1 repetition). Run from `energy_over_time_script/`:
`bash ../energy_over_time_organized/08_exclusion_applied/code/run_mch_fits_no_arbitration.sh /data001/projects/enserrog/AbigailData/energy_over_time_exclude_applied 1 ../energy_over_time_organized/08_exclusion_applied/results`

**Verification of the original pipeline** (`code/PIPELINE_VERIFICATION.md`, `code/reextract.py`, `code/verify_fit.py`): extraction, binarization, energy terms and MCH moments all check out. The excluded reaches were the only data issue found.

**Run log.**
- 2026-10-07, first submission: all 21 tasks failed within seconds. The singularity container cannot see `/data001/projects/...` from these tasks. Output was moved to `results_failed_data001_path_*`.
- 2026-10-07, resubmitted with the filtered inputs copied to `/home/enserrog/ising_EnergyOverTime/data_exclude_applied/` (byte-identical copies). Job ids are in `results/submitted_job_ids.txt`.
- Session dates in `results/` follow the original layout: `<date>_results/<date>_rep1/full_reach/stim_K`.
- 2026-10-07, second submission cancelled. The current committed code (HEAD 5c491cb) has two plotting bugs in `visualization.py` that crash `main.py` after the first stimulus block is fitted, so stim_1/stim_2 are never produced:
  1. `plot_model_quality_summary` prints an undefined `out_path` (NameError).
  2. `plot_energy_across_time` indexes the single-block lists `j_values`, `h_values` and `neural_data` with the stim offset `i` instead of the loop index (IndexError at stim_1).
  The fitting code (`main.py`, `model.py`, `utils.py`) is unchanged since before the July results; the bugs only affect plotting. The July results were evidently produced with a working tree that had these fixed.
- Fix: `code/energy_over_time_script_fixed/` = `git archive HEAD energy_over_time_script` plus those two fixes only (`code/fix_visualization.diff`). Smoke-tested end to end on 210622 (all 3 blocks, per_reach_state written). **The project's own `energy_over_time_script/` is NOT modified.**
- 2026-10-07, third submission from `code/energy_over_time_script_fixed/` (task logs in its `logs/`). Job ids are in `results/submitted_job_ids.txt`.
