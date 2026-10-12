# 03_j_coupling_vs_kinematics_timing

## What this answers

Does the fitted pairwise coupling term J — and energy-derivative spikes — lead, lag, or align with kinematic events (peak acceleration/velocity) during reaching, and how does that timing compare to firing-rate-based signals?

## Scripts

- `arbitration_j_many.py` — Batch, session-by-session test of whether the largest rapid jump in J occurs near peak acceleration/velocity, and whether J leads or lags. Run: `python arbitration_j_many.py --data_folder /path/to/results/`.
- `peak_near_max_velocity.py` — Locates all local J maxima within a window around max velocity for manually specified sessions, with optional Butterworth/Gaussian smoothing. Run: `python peak_near_max_velocity.py --data_folder <dir> --sessions <ids>`.
- `peak_timing_analysis.py` — Reads `pk_j_summary.csv` and ranks which feature (FR peak/trough, J peak, H peak) sits closest in time to peak acceleration. Run: `python peak_timing_analysis.py --csv pk_j_summary.csv --output_dir <dir>`.
- `test_j_timing_vs_kinematics.py` — Compares the J-peak index against nearest acceleration extremum vs. nearest firing-rate extremum, per session/stim. Run: `python test_j_timing_vs_kinematics.py --data_folder <dir> --csv_out <file>`.
- `test_collective_j_timing.py` — Compares J, H-field, firing rate, and Ising energy as candidate signals for indicating the acceleration peak, without requiring the collective-behavior (P(K)) flag. Run: `python test_collective_j_timing.py --data_folder <dir> --csv_out <file>`.

## Key findings

- J-jumps lag the kinematic peak in most sessions (44/63, ~70%) vs. leading in 25% (Rep 1, 21 sessions × 3 stimuli); mean lag ≈ +16.4 to acceleration, +13.6 to velocity — consistent across all three stimuli.
- Despite the lag bias, J vs. firing rate "arbitration" (closeness to acceleration peak) favors J: J wins 35/63 sessions (55.6%) vs. 19 for firing rate, 9 ties — though Stim 2 is near chance (47.6%).
- J's instantaneous correlation with acceleration/velocity is consistently very weak (|r| < 0.15) in both the lag and arbitration analyses, and J–firing-rate correlation is also near zero — the signals appear largely decoupled despite the timing edge.
- Separately, energy-derivative spikes (2.0σ threshold) co-occur with deceleration (pooled Cohen's d = −0.114, p = 0.003) but show no relation to velocity, and an inconsistent, likely confounded relation to position.

## Caveats

- Cross-correlation lag (XClag) signs/magnitudes are inconsistent across stimuli, adding noise to the lead/lag conclusion.
- Position effects in the energy-spike analysis are flagged as a session/trajectory-window confound, not a genuine aggregate signal.
- Multiple-comparisons burden (18 tests) noted in the energy-spike analysis but not fully resolved.
- results/arbitration_J/ and notes/ reflect several iterative rounds of threshold/definition refinement — treat J-jump/peak definitions as version-dependent across scripts.