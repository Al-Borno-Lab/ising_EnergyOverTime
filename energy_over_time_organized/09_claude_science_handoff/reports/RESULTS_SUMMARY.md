# Results Summary — Ising/Maximum-Entropy Analysis of Cerebellar Purkinje-Cell Activity During Reaching

**Source codebase:** `math-alderaan.ucdenver.pvt:/home/enserrog/ising_EnergyOverTime/energy_over_time_script`
(reorganized copy: `energy_over_time_organized/`)
**Basis:** 16 hand-written/generated analysis notes + 27 Python source files, reviewed directly.

## 1. Study design

Pairwise Maximum-Entropy (Ising) models were fit to Purkinje-cell population spike recordings
from mice performing a reaching task, across ~21–68 recording sessions (2021-04-21 through
2022-05-20), each under 3 stimulation conditions (stim 0/1/2) and, in places, multiple
repetitions. The recurring scientific question: does the fitted model's emergent "energy" —
and specifically its pairwise coupling term J — track reach kinematics (position/velocity/
acceleration) better than raw firing rate, and is any such signal a genuine collective
(pairwise-dependent) neural event rather than a firing-rate artifact?

## 2. Energy vs. firing rate — which better predicts the acceleration peak?

*(`02_arbitration_energy_vs_firing/`; source: `notes/Arbitration/accel_results/arbitration_results.md`,
dataset `energy_decomp_may26_stimDecon`, window [390,410], 1 repetition, 63 session×stim combinations)*

- Energy's extremum was closer to the acceleration peak in **30/63 sessions (47.6%)**, firing
  rate in 18/63 (28.6%), with 15/63 (23.8%) ties.
- Energy's win rate was stable across stimulation conditions: Stim 0 = 47.6%, Stim 1 = 52.4%
  (highest), Stim 2 = 42.9% (lowest), each with 5/21 ties.
- The **energy maximum** (rather than energy minimum or either firing-rate extremum) was the
  single most common closest extremum to the acceleration peak.
- Mean distance-to-peak was smaller for energy than firing rate across all three stimuli
  (e.g. Stim 1: 2.810 vs. 3.524) — a modest, consistent edge for energy as a local predictor.
- Energy–firing covariance was small and positive (reported range 0.012–[upper bound
  truncated in source note]).
- **Caveat:** single repetition, single kinematic reference (acceleration); no other note
  exists yet to cross-check or contradict these numbers.

## 3. Does the coupling term J lead or lag movement, and does it out-predict firing rate?

*(`03_j_coupling_vs_kinematics_timing/`; sources: `notes/energy_spike_kinematics_analysis.md`,
`notes/j_analysis/j_kinematics_results.md`, and the Apr-16 report excerpts)*

- J-jumps (largest rapid |dJ/dt|) **lag** the kinematic peak in most sessions — **44/63 (~70%)**
  lag vs. **16/63 (~25%)** lead — with mean lag ≈ **+16.4 samples** relative to acceleration and
  **+13.6 samples** relative to velocity, consistent across all three stimuli. J changes trail
  movement; they do not anticipate it.
- Despite that lag, in head-to-head arbitration against firing rate for closeness to the
  acceleration peak, **J wins 35/63 sessions (55.6%)** vs. firing rate's 19/63 (30.2%), with 9
  ties — though Stim 2 is close to chance (47.6%).
- J's instantaneous correlation with acceleration/velocity is consistently **very weak**
  (|r| < 0.15), and J–firing-rate correlation is likewise near zero: the signals are largely
  decoupled even though J shows a timing edge over firing rate.
- **Energy-derivative spikes** (detected at a 2.0σ threshold, chosen via a 0.5–3.0σ sweep as
  the best power/specificity trade-off) are associated with **deceleration**
  (pooled Cohen's d = −0.114, 95% CI [−0.188, −0.039], **p = 0.003**); velocity showed no
  effect (arguing against a generic arousal/activation confound); the position-kinematics
  association is explicitly flagged by the original authors as a likely session/trajectory-
  window confound rather than a real effect.
- **Caveats:** cross-correlation lag sign/magnitude is inconsistent across stimuli; the
  energy-spike analysis carries an 18-test multiple-comparisons burden that is not fully
  corrected; J-jump/peak definitions evolved across iterative script versions, so numbers
  from different notes are not always directly comparable.

## 4. Are J-peaks genuine collective events, or firing-rate/noise artifacts?

*(`04_collective_activity_pk/`; sources: `notes/pk_analysis/all_sessions_w350_475/SUMMARY.md`
and `outputs/energy_decomp_Jul_10_svg/notes/pk_analysis/pk_hypothesis_scatter_explained.md`)*

Two versions of this test, on different dataset snapshots and thresholds, both show the same
direction with different magnitudes:

- **First version** (21 sessions, 42 stim 0/1 pairs, window bins 350–475, peak threshold 1.75):
  the "collective activity" criterion (Ising-model fit beats independent-neuron fit, with a
  divergence margin) was met in **75% of J-peak sessions vs. 36% of no-peak sessions**
  (χ² = 4.556, df = 1, **p = 0.033**). Mechanistically this arises because the *independent*
  model fits markedly worse during J-peaks (r_independent 0.890 vs. 0.972) — not because the
  Ising fit itself improves (r_ising ≈ 0.96 either way) — i.e., peaks reflect a breakdown of
  the independence assumption, not an improvement in the pairwise model.
- **Second, stricter version** (63 session/stim rows, 39 J-peak vs. 24 no-peak, tolerance-band
  classification rule): **37/39 (94.9%)** of J-peak sessions were "collective" vs. **7/24
  (29.2%)** of no-peak sessions — odds ratio ≈ **44.9**, Fisher's exact **p = 4.3×10⁻⁸**.
- **Caveats:** several detected J-peaks (sessions 210423, 210622, 210511) had near-zero
  Ising/independent divergence, suggesting likely false-positive peak detections; session
  **220515** is a genuine, unexplained exception — strong J peaks and high divergence, yet the
  independent model *out*-fits the Ising model on P(K), possibly due to an unusually strong
  single-neuron (h-field) drive; multiple parallel analysis iterations exist in
  `notes/pk_analysis_jul10/` whose relative currency is not documented.

## 5. Synthetic validation — does a J-jump require a real synchrony effect, or is it a fitting artifact?

*(`06_synthetic_validation_simulation/`; source: `notes/neural_sim/neural_simulation/simulation_writeup.md`)*

- A two-population (100 excitatory "Purkinje" units + 25 interneurons) spiking simulation
  alternated 200 ms blocks of an **independent Poisson** regime (~60 Hz) and a **collective**
  regime (shared two-state Markov burst/quiet dynamics, ~56 Hz) — rates deliberately matched
  so any Ising difference reflects correlation structure, not rate.
- After 5 ms binning, co-firing probability was **~0.41 in burst bins vs. ~0.009 in quiet
  bins** (~45-fold difference); mean per-neuron activation was ≈+0.28 (burst) vs. ≈−0.81
  (quiet).
- An inverse Ising fit (20 subsampled neurons, 210 h/J parameters, MCH/Metropolis MCMC) run on
  this synthetic data supports the interpretation that **a J-jump can arise from correlation
  structure alone, without any change in mean firing rate** — consistent with the real-data
  J-jumps being genuine synchrony signatures rather than firing-rate artifacts.
- **Caveat:** this is a single simulation/note with no independent replication, and by design
  demonstrates plausibility, not a claim about real cerebellar physiology.

## 6. Ising model fitting and phase-transition characterization

*(`01_ising_model_fitting/`)*

No dedicated narrative analysis note exists for this stage — the `results/` tree holds the
underlying quantitative outputs (per-session phase-transition, model-quality, correlation-
order, and energy-distribution files across 21 sessions from 2021-04 through 2022-05, plus
cross-session aggregates), but interpretive write-ups have not been produced from them.
Mechanically: `model.py` fits (h, J) via an iterative Maximum-Entropy solver (default 75
iterations, η=1e-3) and separately sweeps temperature (default range 0.1–2.0, step 0.05) to
locate each session's critical/operating temperature via heat-capacity peaks.

## 7. Decision-tree prediction of arbitration outcomes

*(`05_decision_tree_prediction/`)*

`build_decision_tree_dataset.py` / `run_decision_tree.py` build a per-(session, stim) feature
table (model quality, kinematics, neural statistics) and train a tree/ensemble classifier to
predict whether a session shows a significant J-coupling ("is_j_peak") event, with several
run variants stored in `notes/decision_dataset/`. **No narrative note summarizing outcomes was
found among the reviewed materials** — the run logs/CSVs/plots exist but have not been written
up, so no grounded accuracy or feature-importance numbers can be reported here without
inspecting those run outputs directly.

## 8. Cross-cutting caveats (apply project-wide)

- **Multiple-comparisons burden** is substantial in places (e.g., 18 tests across thresholds/
  signals in the energy-spike-kinematics analysis) and is not always corrected.
- **Dataset-snapshot drift:** several statistics differ slightly between near-duplicate note
  files (e.g., the two P(K) chi-squared/Fisher's-exact results above) because they were
  computed on different dataset snapshots (`energy_decomp_may12_stimDecon` vs.
  `energy_decomp_may26_stimDecon`, etc.) — file dates/paths matter when citing a specific
  number.
- **Session 220515** is flagged in more than one analysis as an unexplained outlier where the
  independent and Ising model fits disagree with the general pattern.
- The **position-kinematics** association is explicitly flagged by the original authors as
  likely a session/trajectory-window confound rather than a real effect — unlike the
  deceleration result, which survived the velocity control.
- The auto-generated "session_report" Markdown files across the project (`notes/session_report/`,
  `outputs/.../reports/`) are **image-gallery indices**, not analysis write-ups — over 90% of
  their content is `<img>`/table markup with no prose; the actual scientific content lives in
  the topic-specific notes summarized above.

## 9. Overall narrative

The project progressed iteratively: (1) build the Ising-model fitting and phase-transition
pipeline; (2) test whether the resulting energy/J signal predicts movement kinematics better
than raw firing rate (finding a real but weak edge for energy/J, with J lagging rather than
anticipating movement, and a modest deceleration-linked energy-spike effect); (3) verify with
a synthetic simulation that a J-jump can arise from correlation structure alone, independent
of firing rate; and (4) build a formal collective-vs-independent statistical test (P(K)
distributions, strongly positive in two independent analyses) plus a decision-tree classifier
intended to move from "is there an effect" toward "can it be predicted from session features"
— the latter's outcome is not yet documented in note form.
