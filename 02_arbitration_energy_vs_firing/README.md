# 02_arbitration_energy_vs_firing/

## What this answers

When does fluctuation in the fitted Ising "energy" of the Purkinje-cell population beat raw firing rate at predicting kinematic events (specifically, acceleration peaks) during reaching, and vice versa?

## Scripts

- `arbitration_many.py` — Batch "arbitration" over all sessions under a data folder: recursively finds per-session `per_reach_state.csv` files, extracts extrema (min/max) of energy and firing rate in a window around a kinematic reference event using multiprocessing, and tests within each session whether energy or firing rate has the extremum closest to that event, producing plots and summary tables. Run from the `energy_over_time_script` directory:
  `python arbitration_many.py --data_folder /path/to/energy_decomp_1/`
  (see `--help` for window size, repetition range, stimulus range, reference signal, output/report directories, and worker count).

## Key findings

Grounded in `notes/Arbitration/accel_results/arbitration_results.md` (data from `energy_decomp_may26_stimDecon`, window [390, 410], reference = acceleration, 1 repetition, stimuli 0–2):

- Across 63 sessions, energy "won" (closer extremum to the acceleration peak) in 30 sessions (47.6%), firing rate won in 18 (28.6%), and 15 (23.8%) were ties.
- Energy win rate was fairly stable across stimulus conditions: Stim 0 47.6% (10/21), Stim 1 52.4% (11/21, highest), Stim 2 42.9% (9/21, lowest), each with 5 ties out of 21 sessions.
- The **Energy Max** was the single most common closest-extremum type (7, 10, and 9 sessions for stim 0/1/2 respectively) — more often than Energy Min, Firing Min, or Firing Max — suggesting energy maxima track acceleration peaks better than energy minima or either firing extremum.
- Mean distance-to-peak was consistently smaller for energy than for firing across all three stimuli (Stim 0: 3.619 vs 3.810; Stim 1: 2.810 vs 3.524; Stim 2: 3.095 vs 4.000), modestly supporting energy as the better local predictor.
- Mean energy–firing covariance was small and positive (reported range 0.012–0.0..., value cut off in source summary).

## Caveats

- The note summary's covariance statistic is truncated in the source material ("0.012–0.") — the upper bound of this range is not available and should be pulled from the original note/results table before being cited.
- Findings are based on a single repetition (rep 1) and one reference variable (acceleration); no cross-repetition consistency or other kinematic references are reported here.
- No other notes exist in `notes/Arbitration/` at this time to cross-check or contradict these results — treat the above as provisional pending replication across reps/sessions.