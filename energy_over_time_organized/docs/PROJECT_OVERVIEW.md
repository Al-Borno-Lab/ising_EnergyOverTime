# Project Overview: Ising/Maximum-Entropy Analysis of Purkinje-Cell Population Activity During Reaching

> **Superseded as the project index by [../README.md](../README.md).** Headline findings below predate the pre-submission checks; several are withdrawn (see RESULTS_LOG_FOR_PAPER.md, table D).

This project fits pairwise Maximum-Entropy (Ising) models to Purkinje-cell population
spike recordings from mice performing a reaching task (~21-68 sessions, 2021-04-21 to
2022-05-20, 3 stimulation conditions each), and asks whether the model's emergent
"energy" and pairwise-coupling term (J) track reach kinematics (position/velocity/
acceleration) better than raw firing rate — and whether any such signal reflects genuine
collective neural structure rather than a firing-rate artifact. The codebase has been
reorganized by scientific question into the numbered folders below, each with its own
README.md.

## 01_ising_model_fitting/
Fits the pairwise Ising model to spike data (Metropolis-Hastings sampling, iterative
Maximum-Entropy learning of h/J) and runs phase-transition analysis (temperature sweeps,
heat capacity) to locate each session's critical/operating temperature. This is the
foundational pipeline everything downstream depends on. See 01_ising_model_fitting/README.md
for details.

## 02_arbitration_energy_vs_firing/
Asks whether Ising "energy" or raw firing rate falls closer to the kinematic (acceleration)
peak. Headline finding: energy wins in 30/63 sessions (47.6%) vs. 18 for firing rate (15
ties) — energy maxima were the most common closest extremum, a modest but real edge.
See 02_arbitration_energy_vs_firing/README.md for details.

## 03_j_coupling_vs_kinematics_timing/
Asks whether the J-coupling term (and energy-derivative spikes) lead or lag kinematic
events, and re-tests J against firing rate/H-field/energy as competing predictors.
Headline findings: J beats firing rate in arbitration (35/63 vs. 19 sessions) but both
correlate only weakly with kinematics (|r|<0.15); J lags the kinematic peak in ~70% of
sessions (mean lag ≈ +16.4 samples) — J changes trail movement rather than anticipate it;
energy-derivative spikes (2.0σ threshold) are associated with deceleration (pooled Cohen's
d = -0.114, p = 0.003), with no velocity effect and a position effect judged confounded.
See 03_j_coupling_vs_kinematics_timing/README.md for details.

## 04_collective_activity_pk/
Tests whether J-peak sessions reflect genuine pairwise-dependent (collective) neural
structure by comparing empirical, independent, and Ising-model spike-count [P(K)]
distributions. Headline finding: J-peak sessions are far more likely to show real
collective structure than no-peak sessions (one version: χ²=4.556, p=0.033; a related
note: Fisher's exact OR≈45, p=4.3×10⁻⁸) — J peaks arise because the independent-neuron
model fails, not because the Ising fit itself improves. See
04_collective_activity_pk/README.md for details.

## 05_decision_tree_prediction/
Builds a per-(session, stim) feature table (model quality, kinematics, neural stats) and
trains a decision-tree/ensemble classifier to predict `is_j_peak` from measurable session
features, moving from "is there an effect" toward "can it be predicted." See
05_decision_tree_prediction/README.md for details.

## 06_synthetic_validation_simulation/
A synthetic two-population (excitatory/inhibitory) simulation alternating independent
("random") and correlated ("collective") firing regimes at matched mean rates (~56-60 Hz).
Headline finding: J can jump during collective epochs from correlation structure alone,
with no rate change — supporting the interpretation of real-data J-jumps as genuine
synchrony signatures rather than firing-rate artifacts. See
06_synthetic_validation_simulation/README.md for details.

## shared/, docs/, logs/
`shared/` holds the cross-cutting `src/` package, batch/aggregation scripts (cross-session
meta-analysis: inverse-variance-weighted Cohen's d, sign tests, Fisher's combined p-value),
utility scripts, and reference/aggregate data used across folders 01-06. `docs/` contains
this overview plus the auto-generated per-session image-gallery reports
(`docs/session_reports/`) — these are figure-grid indices for browsing plots, not analysis
write-ups; the actual scientific narrative lives in each folder's README. `logs/` holds
SLURM/Ising job logs (master, tasks, single-test); neural-simulation logs have been moved
into `06_synthetic_validation_simulation/`.

## Known caveats across the project
- **Multiple comparisons:** several analyses (e.g., the spike-kinematics threshold sweep,
  18 tests across thresholds/signals) are not always corrected for multiple comparisons —
  treat individual p-values cautiously.
- **Dataset-snapshot inconsistencies:** near-duplicate notes report slightly different
  statistics (e.g., two versions of the P(K) chi-squared/Fisher's-exact result) because
  they were run on different dataset snapshots (e.g., `energy_decomp_may12_stimDecon` vs.
  `energy_decomp_may26_stimDecon`) — check file dates/paths before citing a specific number.
- **Unexplained outlier:** session 220515 is flagged as an outlier where independent and
  Ising model fits disagree with the general pattern.
- **Confounded result:** the position-kinematics finding is explicitly flagged by the
  original authors as likely a session/trajectory-window confound rather than a real
  effect, unlike the deceleration result.