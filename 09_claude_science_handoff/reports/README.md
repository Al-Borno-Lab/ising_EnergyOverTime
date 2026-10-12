# 08_exclusion_applied: MCH refit with Dylan's excluded reaches removed (started 2026-10-07)

**Why.** Dylan's source files (`/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_*_NPresults_short.mat`) flag 157 of the 1249 analysed reaches as `ReachS.exclude = 1`. The original `*_bin10_spike.mat` files predate the exclude export, so every flagged reach was used in `01_ising_model_fitting/results`. Decision: drop them, refit, and report both versions.

**Inputs.** `/data001/projects/enserrog/AbigailData/energy_over_time_exclude_applied/*_bin10_spike.mat`: copies of the originals with flagged reaches removed (1249 → 1092 reaches; the smallest block keeps 8). Built by `code/make_excluded_copies.py` and reloaded with `utils.loadmat` as a check. Per-block counts are in `code/exclusion_filter_log.csv`.

**Fitting.** `code/run_mch_fits_no_arbitration.sh` is byte-identical to `energy_over_time_script/process_matlab_file_manyTimes_sbatch.sh` up to `rm job_template.sh`. The arbitration follow-up is removed. Same `main.py` arguments (MCH, max_iter 200, eta 0.05, sample_size 1e5, rows 100:800, full_reach, 1 repetition). Run from `energy_over_time_script/`:
`bash ../energy_over_time_organized/08_exclusion_applied/code/run_mch_fits_no_arbitration.sh /data001/projects/enserrog/AbigailData/energy_over_time_exclude_applied 1 ../energy_over_time_organized/08_exclusion_applied/results`

**Verification of the original pipeline** (`code/PIPELINE_VERIFICATION.md`, `code/reextract.py`, `code/verify_fit.py`): extraction, binarization, energy terms and MCH moments all check out. The excluded reaches were the only data issue found.
