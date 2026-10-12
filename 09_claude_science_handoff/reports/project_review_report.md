# Project Review: `ising_EnergyOverTime/energy_over_time_script`

**Location:** `math-alderaan.ucdenver.pvt:/home/enserrog/ising_EnergyOverTime/energy_over_time_script`
**Reviewed:** notes/ (16 markdown files), src/ + top-level pipeline scripts (27 Python files), plus generated report skeletons.

## 1. What the project is

This is a research pipeline that fits **pairwise Maximum-Entropy (Ising) models** to
Purkinje-cell population spike recordings from mice performing a reaching task, and asks
whether the model's emergent "energy" (and its coupling term J) tracks reach kinematics
(position/velocity/acceleration) better than raw firing rate does. It spans ~21-68
recording sessions (dated 2021-04-21 through 2022-05-20), each with 3 stimulation
conditions (stim 0/1/2) and (in places) multiple repetitions, stored under a data root
referred to as `AbigailData/energy_over_time/energy_decomp_*`.

## 2. Core pipeline (main.py / model.py / analysis.py / visualization.py / utils.py)

- **`utils.py`** — loads `.mat` files (including HDF5-backed MATLAB v7.3 via h5py fallback),
  preprocesses spikes, checks perturbation regime, computes firing rates and CIs.
- **`model.py`** — the Ising engine: `calc_e` computes configuration energy from (h, J);
  `metropolis` runs Metropolis-Hastings sampling at a given temperature; `fit_ising_model`
  learns (h, J) via an iterative Maximum-Entropy solver (gradient-style learning, default
  75 iterations, η=1e-3, seeded RNG); `phase_transition_analysis` sweeps temperature
  (default 0.1–2.0, step 0.05) computing avg spin, energy, and **heat capacity** to locate
  each session's critical/operating temperature; `learn_settings` configures the fit.
- **`analysis.py`** — downstream analysis on fitted models: model-quality evaluation,
  per-stimulus energy analysis, transition-point / firing-rate-transition-point detection,
  energy↔temperature mapping (`energy_to_temp`), trial-level statistics.
- **`visualization.py`** — all plotting: phase-transition curves, energy-across-time,
  transition-point overlays, energy histograms/distributions by K, and the multi-panel
  "model quality summary" figure (A–D panels: correlation orders, K-distributions, etc.).
- **`main.py`** (canonical entry point) / **`main-wells.py`** / **`main-wells-old.py`** —
  variants tying the above together end-to-end for different data sources (raw `.mat`
  files vs. pre-aligned Wells-lab JSON), each: load → fit Ising model → phase-transition
  scan → energy calculation → visualize → write CSVs. `main.py` is the actively-maintained
  version (has extra `--energy_by_k_samples`/`--skip_energy_by_k` flags not in the
  Wells variants).
- **`src/processing.py`, `src/util.py`** — shared helpers (session averaging, extrema
  finding, confidence intervals, recursive file discovery).

## 3. The central scientific question and how it's tested

Three interlocking questions recur across the batch-analysis scripts and notes:

1. **Does Ising "energy" track kinematics better than firing rate?**
   (`arbitration_many.py`, `src/arbitration.py`, `notes/Arbitration/...arbitration_results.md`)
   — per-session test of which signal's extremum falls closer to the acceleration peak.
2. **Does the coupling term J jump/peak near movement onset, and does it lead or lag?**
   (`arbitration_j_many.py`, `peak_near_max_velocity.py`, `peak_timing_analysis.py`,
   `pk_j_analysis.py`, `test_j_timing_vs_kinematics.py`, `test_collective_j_timing.py`)
   — J-jump index = argmax|dJ/dt| in a search window; lag = (J-jump − kinematic-peak index).
3. **Are J-peaks/energy-derivative spikes genuine collective (pairwise-dependent) neural
   events, or noise?** (`pk_j_analysis.py`'s P(K) comparison of empirical vs. independent
   vs. Ising-model spike-count distributions; `build_decision_tree_dataset.py` +
   `run_decision_tree.py` train a classifier for `is_j_peak`.)

### Key quantitative findings from the notes

- **Arbitration (energy vs. firing, accel peak):** energy won 30/63 sessions (47.6%),
  firing 18, ties 15; energy maxima were the most common closest extremum.
- **Arbitration (J vs. firing):** J won 35/63 sessions (55.6%) vs. 19 for firing rate —
  a modest edge, but J↔firing and J↔kinematics correlations were both very weak (|r|<0.15).
- **J-jump timing:** J lags the kinematic peak in ~70% of sessions (44/63), leads in ~25%
  (16/63); mean lag ≈ +16.4 samples relative to acceleration — J changes trail movement,
  they don't anticipate it.
- **Energy-derivative spikes vs. kinematics:** using a 2.0σ threshold (chosen via a
  0.5–3.0σ sweep as the best power/specificity trade-off), spikes are associated with
  **deceleration** (pooled Cohen's d = −0.114, 95% CI [−0.188, −0.039], p = 0.003);
  velocity showed no effect (ruling out a generic arousal/activation confound); the
  position result was judged a session-specific confound, not a real effect.
- **Collective-activity test (P(K) analysis):** J-peak sessions are far more likely to show
  genuine pairwise-dependent structure than no-peak sessions — one version reports
  χ²=4.556, p=0.033 (75% vs. 36% "hypothesis met"); a related note reports an even stronger
  Fisher's-exact result (37/39 vs. 7/24 sessions, odds ratio ≈45, p=4.3×10⁻⁸) — i.e., J
  peaks arise mainly because the *independent*-neuron model fails to explain the spike-count
  distribution during those windows, not because the Ising fit itself improves.
- **Synthetic validation (`simulate_neural_regimes.py` / `notes/.../simulation_writeup.md`):**
  a two-population (excitatory/inhibitory) simulation alternating "random" (independent
  Poisson) and "collective" (shared burst/quiet Markov) firing regimes — matched to nearly
  identical mean rates (~56-60 Hz) — confirms that J can jump during collective epochs from
  correlation structure alone, without any rate change, supporting the interpretation of the
  real-data J-jumps as genuine synchrony signatures rather than firing-rate artifacts.

## 4. Supporting infrastructure

- **Batch/aggregation layer:** `aggregate_transition_analysis.py` (1334 lines) and
  `aggregate_model_quality_plots.py` scan a results tree, group sessions by
  stim/phase label, and run cross-session meta-analysis (inverse-variance-weighted
  Cohen's d, sign test, Fisher's combined p-value, threshold sweeps) — this is the engine
  behind the "pooled" statistics quoted above.
- **Reporting:** `generate_session_report.py` + `compare_sessions.py` auto-generate the
  large Markdown "session report" files seen in `notes/session_report/` and
  `outputs/.../reports/` — these are **image-gallery indices** (figure grids per session),
  not written narratives; most contain no prose (confirmed by stripping markup: >90% of
  their content is `<img>`/table markup). The one exception is
  `energy_decomp_Apr_16_report.md`, which has a later hand-written "J-Jump vs Kinematic
  Peak Analysis" section appended to an otherwise-empty gallery skeleton.
- **Decision-tree layer:** `build_decision_tree_dataset.py` builds a per-(session,stim)
  feature table (model quality, kinematics, neural stats) with target `is_j_peak`;
  `run_decision_tree.py` trains/visualizes a tree or ensemble classifier with CV,
  feature importance, and confusion-matrix plots.
- **Utility scripts:** `collect_files.py` (recursive copy-by-pattern), `regen_model_quality_summary.py`
  (regenerate figures without refitting), plus several MATLAB-preprocessing and SLURM
  `sbatch` submission shell scripts for running the pipeline on the cluster.
- **Testing:** `test_j_timing_vs_kinematics.py` and `test_collective_j_timing.py` compare
  J-coupling timing against firing rate/H-field/energy as competing predictors of the
  acceleration peak, with gating on the "genuine collective event" flag from the P(K)
  analysis (question 3 above) — these look like the most recent, most rigorous iteration
  of the arbitration question, built to arbitrate between all candidate signals at once
  rather than just J vs. firing rate.

## 5. Overall narrative

The project has progressed through iterative refinement: (a) build the Ising-model fitting
and phase-transition pipeline (`main.py`/`model.py`), (b) ask whether the resulting energy/J
signal is a better predictor of movement kinematics than raw firing rate (arbitration
scripts), (c) find a real but weak/noisy signal — J lags movement and weakly favors
deceleration — that survives a battery of statistical controls (multiple thresholds, cross-
session pooling, sign tests), (d) verify with a synthetic simulation that a J-jump can arise
from correlation structure alone, and (e) build a formal collective-vs-independent test
(P(K) distributions) plus a decision-tree classifier to move from "is there an effect" to
"can it be predicted/explained by measurable session features." The auto-generated
"session_report" Markdown files are navigational figure indices for browsing all sessions'
plots, not analysis write-ups — the actual scientific content lives in the `notes/*.md`
prose files summarized in §3.

## 6. Caveats noted directly in the notes

- Multiple-comparisons burden is substantial (e.g., 18 tests across thresholds/signals in
  the spike-kinematics analysis) and not always corrected.
- Several statistics differ slightly between near-duplicate note files (e.g., two versions
  of the P(K) chi-squared/Fisher's-exact result with different p-values) — these appear to
  be sequential re-runs on slightly different dataset snapshots (`energy_decomp_may12_stimDecon`
  vs. `energy_decomp_may26_stimDecon` etc.), so file dates/paths matter when citing a number.
- At least one session (220515) is flagged as an unexplained outlier where independent
  and Ising model fits disagree with the general pattern.
- The position-kinematics result is explicitly flagged by the authors as likely a
  session/trajectory-window confound rather than a real effect.
