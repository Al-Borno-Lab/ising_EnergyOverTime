# Handoff: Ising / maximum-entropy analysis of Purkinje-cell populations (status 2026-10-07)

This file is everything needed to continue the project in Claude Code. Read sections 1–3 first, then 7–9.
Every artifact from the Claude Science sessions is in `ising_PC_project_handoff.zip` (layout in section 4).

---

## 1. The project in one paragraph
Pairwise maximum-entropy (inverse Ising) models are fitted to Neuropixels recordings of mouse cerebellar Purkinje cells (PCs) during a reach task, using the Calame, Becker & Person dataset (bioRxiv 10.1101/2021.12.17.473247; PDF in `papers_and_pdfs/`). Each session has 3 blocks: `stim_0` baseline, `stim_1` closed-loop optogenetic stimulation triggered when the hand crosses 1 cm outward (stimulation parameters in the dataset paper's Methods; whether all 6 mice are opsin-positive is an open question for the lab), and `stim_2` washout. There are 21 sessions from 6 mice: fChR2, fChR4, fChR5, cChR1 (2021) and DJC000, DJC002 (2022, not in the preprint), with 3–32 PCs each. Goal: **one paper** on what the Ising model reveals about PCs that other methods cannot.

## 2. Where everything lives

### Cluster: `math-alderaan.ucdenver.pvt` (user `enserrog`, **campus VPN required**)
| What | Path |
|---|---|
| Project git repo | `/home/enserrog/ising_EnergyOverTime` (the remote is not ours: commit locally, don't push unless asked) |
| Original pipeline code | `/home/enserrog/ising_EnergyOverTime/energy_over_time_script/` (`main.py`, `model.py`, `utils.py`, `visualization.py`, `analysis.py`, `arbitration_*`, `process_matlab_file_manyTimes_sbatch.sh`) |
| Question-organized tree | `/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/` (folders `01_ising_model_fitting` … `08_exclusion_applied`, `shared/`, `RESULTS_LOG_FOR_PAPER.md`) |
| July 2026 MCH fits (all reaches) | `energy_over_time_organized/01_ising_model_fitting/results/<date>_results/<date>_rep1/full_reach/stim_K/` |
| **MCH refit with excluded reaches dropped (current primary)** | `energy_over_time_organized/08_exclusion_applied/results/` (63/63 blocks complete) |
| Fixed copy of the pipeline used for that refit | `08_exclusion_applied/code/energy_over_time_script_fixed/` (git commit `ddf8de9`; diff `code/fix_visualization.diff`) |
| Pre-submission checks, patches, time-varying-field outputs | `energy_over_time_organized/07_presubmission_checks/` (`patches/`, `timevarying_field/`, `regenerated/`) |
| **True raw source** (Dylan's files) | `/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_<date>_<mouse>_NPresults_short.mat` (24 files, MATLAB v7.3/HDF5, read with h5py) |
| Files the pipeline consumes | `/data001/projects/enserrog/AbigailData/energy_over_time/<date>_<mouse>_bin10_spike.mat` (21 files) |
| Same, with `exclude==1` reaches removed | `/home/enserrog/ising_EnergyOverTime/data_exclude_applied/` (must be under $HOME, see gotchas) |
| Singularity image (coniii + Boost sampler) | `~/projectDir/singularity-env/inverse-ising-arm-2.sif` |

### Cluster facts
- **SLURM:** account `ceas-alborno`. Partitions: `math-alderaan` (the pipeline's), `math-alderaan-short`, `math-alderaan-quick`, `math-alderaan-gpu*`, `jupyter`.
- **Nodes:** 64 CPUs and roughly 256–512 GB RAM. Request many cores (`--cpus-per-task=16..64`) and parallelize inside the job; past one-core jobs under-used the hardware.
- **Running Python:** `singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python script.py ...`. The container's Python is **3.10**.
- **Login node:** do not run heavy work on it; always `sbatch`. Its `python3` has **no numpy** (fine for pure-python summaries).
- **MCH pipeline settings** (`job_template.sh` inside the master script): `main.py --bin_size 1 --sample_size 100000 --n_cpus 64 --max_iter 200 --eta 0.05 --temp_min 0.1 --temp_max 2.0 --temp_step 0.05 --metropolis_samples 100000 --truncate_idx_l 100 --truncate_idx 800 --confidence 0.8 --firing_rate_window 1`. Each job uses 64 CPUs and 256 GB for up to 12 h, one job per file. Run from `energy_over_time_script/` (or the fixed copy) with `bash <master>.sh <input_dir> 1 <output_dir>`. The no-arbitration variant is `08_exclusion_applied/code/run_mch_fits_no_arbitration.sh`.

### Cluster gotchas (each cost a failed run)
1. **The container can't see `/data001/projects/...` in the user's own sbatch jobs.** Tasks fail in seconds with `FileNotFoundError`. Copy inputs under `$HOME` first.
2. **`visualization.py` at HEAD (`5c491cb`) crashes `main.py` after stim_0**, leaving stim_1 and stim_2 missing:
   - (a) `plot_model_quality_summary` prints an undefined `out_path`;
   - (b) `plot_energy_across_time` indexes `j_values`/`h_values`/`neural_data` with `i` instead of `_loop_i`.
   Use the fixed copy, or apply `fix_visualization.diff`. The fitting code is unaffected.
3. **Python 3.10 in the container rejects backslashes inside f-string `{}`**, which breaks `pk_j_analysis.py`; see `patches/02_pk_j_analysis_fstring.diff`.
4. **Scripts in `03_.../code` need `shared/` and `04_.../code` on `sys.path`**: patches `03a`/`03b`, or set `PYTHONPATH`.
5. **`aggregate_transition_analysis.py` `session_label()` returns `full_reach` for every session** in the organized layout: patch `01`.
6. **Never `pip install coniii`.** The PyPI builds fail or lack the Boost sampler; use the container.
7. **The master script writes and deletes `job_template.sh` in the cwd.** Don't run two masters in the same directory at once.
8. **The four patches in `07_presubmission_checks/patches/` are tested but NOT applied** to the user's code. Applying them is the user's decision.

## 3. Data conventions (verified end to end from Dylan's raw files)
- **Bins:** 10-ms bins (1 sample = 10 ms). Analysis window = rows 100:800 of each reach segment (T = 700 bins).
- **Spikes:** `Bin10` holds spike counts; the pipeline binarizes them as count > 0. About 21% of non-empty bins contain ≥ 2 spikes. Spins are `s = 2x − 1 ∈ {−1, +1}`.
- **Alignment:** **bin 400 = the 1-cm outward crossing = the stimulation trigger**. Segments are centred on it by construction; laser onset falls at a median bin of 400.1 (IQR 397.8–400.9). Per-reach "peak velocity" taken over the whole window is NOT a reliable anchor (the maximum often lands far from the reach); use bin 400 or the condition-level landmarks.
- **Kinematics:** the kinematics in `bin10_spike.mat` / `per_reach_state.csv` are **min-max scaled per reach**. Unscaled hand position (cm) is in `ReachS.filt_kin` / `real_kin` of the raw files, and laser times are in `ReachS.stimtime`.
- **Excluded reaches:** `ReachS.exclude == 1` flags 157 of 1249 reaches (lower `qm`). They were **never removed** in the July pipeline. The user decided to drop them and report both versions; the primary analysis now has 1092 reaches.
- **Energy:** E(s) = −(Σ hᵢsᵢ + Σ_{i<j} Jᵢⱼsᵢsⱼ). `J_parameters.csv` is one row in upper-triangular order. `per_reach_state.csv` columns: `reach_idx, stim, x, y, z, firing_rate, energy, j (=ΣJss), h (=Σhs), spike_n0..`.
- **Unused raw files:** `210414_WT2` (old format), and `220519_DJC000` / `220520_DJC002` (no `ReachS.stim`).

## 4. Zip layout (`ising_PC_project_handoff.zip` → `handoff/`)
| Folder | Contents |
|---|---|
| `code/` | All sandbox code (see section 5), the patch diffs, and copies of a few cluster scripts (`03_...__arbitration_j_many.py`, etc.; `__` stands for `/` in the original path) |
| `reports/` | Every markdown report. Start with **`COLLABORATION_BRIEF.md`**, **`RESULTS_LOG_FOR_PAPER.md`**, **`PREREGISTRATION_OPERATING_POINT.md`**, `PAPER_ANALYSIS_PLAN.md`, `PIPELINE_VERIFICATION.md`, `TIMEVARYING_FIELD_ISING_REPORT.md` |
| `figures/` | All PNG figures, including `h0_synthetic_validation.png`, `rates_vs_couplings_heldout.png`, `fig5c_field_vs_firing_rate.png` and `tvfield_vs_independent.png` |
| `papers_and_pdfs/` | `ISING_FINDINGS_FIGURES.pdf` (15 captioned figures), the dataset paper, Nashef 2023, Tkačik 2014, and others |
| `tables/` | Every CSV/JSON result table |
| `data/` | `spikes_v2.npz` (**the main local data file**), `tvfield_cv_raw_excl.pkl`, `h0_truth_J.npz`, `per_reach_cosilence.csv.gz` (the per-bin compact table of the July fits), etc. |
| `artifact_list.csv` | Filename → folder mapping |

## 5. Sandbox code inventory (`code/`)
**Python env:** numpy, scipy, pandas, matplotlib, statsmodels; no coniii needed.

| File | Purpose |
|---|---|
| `tvfield_model.py` | Time-varying-field pairwise model fitted by **pseudo-likelihood**. `bspline_basis(400)` gives cubic B-splines with 50-ms knots from −600 to +1600 ms and 250-ms knots outside. `fit(X, B, model)` supports models `M0` (time-varying independent), `M1` (time-varying fields + fixed J), `Ms` (static) and `Mlag` (200-ms-lag control). Also `pseudo_ll`, `exact_ll`, `trial_shuffle`. **Do not use its parameters for thermodynamics**: plain pseudo-likelihood fails moment-matching in the strongly coupled 2022 sessions |
| `thermo_model.py` | **Exact maximum likelihood** (moment matching) with time-varying fields + fixed J, by enumeration for N ≤ ~20. Functions: `fit_exact`; `fit_exact_epoch` (J + D in a mask window); `loglik_bins`; `fit_fields_only`; `thermo` (per bin: entropy, multi-information I, χ_J, heat capacity, per-neuron conditional MI, synchronous-pause tails); `sample` (exact sampling); `timelocked_cov`. Unit-tested against brute force to 1e-15 |
| `run_tvfield.py` | 5-fold CV of M0/M1/Mlag/Ms plus a 20-shuffle null over 63 conditions. `python run_tvfield.py spikes_v2.npz phase_landmarks.csv 20 12 drop_excluded out.pkl`. About 1 h on 12 cores. Produced `tvfield_cv_raw_excl.pkl` |
| `run_static_ll.py` | Exact held-out log-likelihood of the static Ising and static independent models on the same folds, N ≤ 20. About 2 min. Its output `static_ll_excl.pkl` was **not** kept, so rerun it if needed |
| `run_h0.py` | H0 synthetic validation (100 tasks: scenarios A–E × N ∈ {11, 15} × R ∈ {20, 60} × 5 replicates). About 2 h on 12 cores |
| `run_epochJ.py`, `run_pruned.py` | Epoch-specific-J test; refit after pruning pairs with r > 0.5 |
| `export_spikes_v2.py` | Builds `spikes_v2.npz` from the bin10 files (runs on the cluster). Keys per condition `k = "<date>_<stim>"`: `X_k` (R, 700, N) binary; `C_k` counts; `t_k` bin times; `reachnum_k`; `fold_k` (5 folds, rng seed = stim); `exclude_k` (Dylan's flag). **Always filter with `exclude_k == 0`** |
| `reextract.py`, `inspect_src.py`, `verify_fit.py` | Raw-file → bin10 → fit verification. Everything matched exactly; see `PIPELINE_VERIFICATION.md` |
| `make_excluded_copies.py` | Writes bin10 copies with excluded reaches removed |
| `build_findings_pdf.py` | Regenerates `ISING_FINDINGS_FIGURES.pdf` (expects `fig_paths.json`) |
| `make_patches.py` + `*.diff` | The 4 tested-but-unapplied code patches |

`phase_landmarks.csv` (in `tables/`) holds per-condition onset / peak_acc / zero_cross (peak velocity) / peak_dec / decel_end bins. Median peak velocity is at bin 400, onset −90 ms, peak acceleration −40 ms.

## 6. Standing conventions (user preferences)
- **One results log:** keep a single paper-facing log, `RESULTS_LOG_FOR_PAPER.md`, and update it rather than replacing it. Log only results we are sure of.
- **Plain language** when asked ("explain as if I was 25").
- **Statistics:** the unit is the **session** (n = 21; condition values averaged within a session), Wilcoxon signed-rank tests, BH-FDR. Also report cohort (2021 vs 2022) and treat mouse (6 mice) as a random effect.
- **Timing claims** (when an extremum occurs) must be compared against **control windows at rest**.
- **Synthetic data:** use it only for method validation, clearly labelled. "Before making up any data, I will get the raw data."
- **Saving and git:** save all code as soon as it works. On the cluster, commit completed steps to the git repo, **scoped to that step's paths only**. Never sweep in the user's uncommitted edits: there are modified files in `energy_over_time_script/` and untracked folders.
- **Pre-registration:** the operating-point test has one (`PREREGISTRATION_OPERATING_POINT.md`). Any change before real-data results must be logged as a dated amendment.

## 7. What is established
1. **Collective structure:** the stationary Ising model reproduces the population statistics. Data, Ising and independent P(K) agree for the Ising fit and not the independent one: the Ising wins in 62/63 blocks, with the independent model's error about 2.2× larger. Real |J| exceeds 100 rate-preserving shuffle refits in 63/63 conditions.
2. **The stationary model's "J rises during the reach" is an artefact.** Shared rate changes leak into ΣJss. With time-varying fields (M1), the couplings still improve held-out prediction in **21/21 sessions** (62/63 conditions above the trial-shuffle null with exclusions applied; exact likelihood agrees). That coupling is present at rest, with **no detected change during the reach** (95% CI on the change −28% to +93%).
3. **Rates dominate during the reach; couplings help at all times** (`rates_vs_couplings_heldout.png`; exact held-out likelihood, 13 sessions with N ≤ 20).
   - In the ±200-ms reach window, time-varying independent beats static Ising in 11/13 sessions (p = 0.15).
   - At baseline, static Ising is better.
   - Over the whole trial they tie (6/13).
   - The time-varying Ising beats both (vs time-varying independent: 12/13 sessions, p = 0.0005).
   - **Do not claim "time-varying independent beats static Ising" in general.**
4. **Field-energy dip at peak acceleration** (`fig5c_field_vs_firing_rate.png`). The time-varying field term −Σhᵢ(t)sᵢ troughs at a median of −40 ms from peak velocity (= median peak acceleration). It falls in the 200 ms before peak velocity in 60% of conditions vs 14% of control windows; it is 15 vs 3.6 baseline-SD deep, in 20/21 sessions (p = 2e-6). The population-average firing rate barely moves, because neurons change in both directions.
5. **Exclusion sensitivity:** dropping the 157 flagged reaches changes nothing material. MCH J correlates with the July fits at median 0.996, slope 1.00; the time-varying-field conclusions are unchanged.

## 8. The current novel hypothesis and its status
**Operating-point hypothesis** (see `PREREGISTRATION_OPERATING_POINT.md` v3, `PAPER_ANALYSIS_PLAN.md`, `COLLABORATION_BRIEF.md`): with J fixed, the reach moves PCs toward p = 0.5, so the fixed couplings gain influence (multi-information, susceptibility) during the reach.

- **H0 synthetic validation: DONE** (`h0_synthetic_validation.png`, `h0_results.csv`).
  - The primary measure (reach-window ΔI/N from the exact fixed-J model) is accurate: sign correct 40/40, within 50% of truth 40/40, null ≈ 0, J recovery r = 0.96–0.99.
  - Three secondary tests failed calibration and were amended (amendments 2–4):
    - per-pair epoch-J is underpowered → **single coupling scale κ**, J_eff = (1 + κ)J;
    - the neuron-level ρ is biased when there is no coupling → **ρ_real − ρ_null**;
    - the pause-tail metric is non-monotone → H5 becomes the silent-count variance (= χ_J/4) and is **dropped from the primary family**.
  - The κ test and the null-corrected neuron test still need a short synthetic check (scenarios A, C, D, E) before real data.
- **H1 (operating point, real data): computed, and it goes AGAINST the premise.**
  - Reach window vs baseline: 60% of 1092 neurons move **away** from 0.5.
  - Session-mean p(1 − p) **decreases** in 17/21 sessions (p = 0.018).
  - Fixed J therefore predicts that coupling influence **falls** during the reach, which the pre-registration counts as refutation.
  - The likely replacement claim: *during acceleration PCs polarize (commit to firing or silence), so the population becomes more rate-driven and less coupling-driven while J stays unchanged.* This is consistent with result 3 and with the field-dip timing in result 4.

## 9. Next steps (in order)
1. Synthetic check of amendments 2–3 (κ test; ρ_real − ρ_null) with `thermo_model.py` + `run_h0.py` (reuse the scenarios). Log the outcome in the pre-registration.
2. **Run H2–H4 on real data as pre-registered.** Use exact fits for N ≤ 20, and 10 random 16-neuron subsets for N > 20 (amendment 1). Run on the **cluster**: exact fits at N = 17–19 with 20-shuffle nulls for 63 conditions are too heavy locally. Per condition:
   - fit the time-varying-field model + fixed J exactly;
   - compute `thermo()`;
   - compute ΔI/N, Δχ_J/N and Δv in the reach window (bins 380–420) vs baseline (300–360);
   - run 20 trial-shuffle refits;
   - fit κ;
   - compute the empirical time-locked covariance.
   Cost: about 35 s per fit and 12 s per `thermo` at N = 15 with 60 reaches.
3. Whichever way H2 goes, update `RESULTS_LOG_FOR_PAPER.md` and `COLLABORATION_BRIEF.md`.
4. Later: the stimulation test with true laser times (`ReachS.stimtime`), testing whether the pulse changes κ in the pulse window; a bin-size sweep (5/20 ms) against Nashef et al. 2023; unit positions (`cellData.channel/depth`) for spatial structure.
5. **Questions for the lab:**
   - how were the 2022 sessions (DJC000/DJC002) recorded and sorted?
   - was the unit-stability filter applied (the paper kept 126/231 units)?
   - what exactly do `exclude` and `qm` mean?

## 10. Earlier work (context only)
- The literature review (`literature_review_coupling_methods.md`) covers coupling-extraction methods and correlated-activity methods.
- Comparisons with Tkačik 2014 and Nashef 2023.
- Phase-timing analyses: the old "J peaks after the reach" claim was withdrawn as a window artefact.
- The flat-firing-rate J dissociation is rare (`FLAT_FR_J_DISSOCIATION.md`).
- The stimulation-block first look is preliminary: per-block fits, nothing survives FDR.
- Stimulation-model code (`stimfit_*`) was lost and must be rewritten from its design: shared J, block-specific spline fields, per-group/window intercepts, a scalar κ in pulse and control windows. It was validated earlier: κ recovered at 0.50–0.53 for a true 0.5 change, and about 0 under rate-only effects.
