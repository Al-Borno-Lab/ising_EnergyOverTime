# Current results status (as of 2026-09-24)

Project tree: `math-alderaan.ucdenver.pvt:/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/`
Data: 21 sessions x 3 stimuli = 63 conditions (raw .mat available for 9 sessions only).

## 1. Established (independently verified)

| Claim | Evidence | Source |
|---|---|---|
| Pipeline from raw spikes to energy, J-term, h-term is correct | Bit-exact reproduction from scratch for all 9 sessions with raw .mat (27 session x stim) | VERIFICATION_REPORT.md |
| Purkinje populations show collective structure beyond independent firing | Model-free surrogate test (500 rate-preserving shuffles; L1 distance of P(K) to independent-Bernoulli null) rejects null in 27/27 conditions, z = 4.9 to 116, p at resolution floor (~0.002) | falsification_findings.md |
| J can change from correlation structure alone, without a rate change | Synthetic two-population simulation | 06_synthetic_validation_simulation |

Scope limit: the falsification test shows collective structure exists; it does not validate that the pairwise Ising model is the right description of it. The surrogate re-fit check (fit J on the same surrogates and confirm it collapses toward zero) is still open.

## 2. J-coupling timing vs. reach kinematics

- Peak-to-peak test (verification_j_timing_accel_vs_decel.md): no configuration shows a significant excess of J-peaks in the acceleration phase. Sign tests are never significant; Wilcoxon on lag magnitude is nominally significant in 5 of 8 configurations, always in the deceleration direction (J peaks after peak velocity). Collective + J-peak reaches: 15 acceleration vs 21 deceleration, median lag +1 sample, Wilcoxon p = 0.035, sign p = 0.405. Per-session (n = 14): median lag +19.75, Wilcoxon p = 0.039.
- Interpretation: collective coupling co-occurs with the reach but leans toward the deceleration phase. This does not reproduce an acceleration-locked framing from PMC10370034.
- These tables used direct-subtraction lags, so they are not affected by the `_xcorr_lag` sign bug; re-confirmation on regenerated outputs is still recommended.

## 3. Time-resolved fluctuation (criticality proxy) pilot, sliding-window variance

Per-session extremum time minus kinematic peak (positive = after), n = 20 to 21 sessions, BH-FDR corrected:

| Trace | Extremum | vs velocity: leads/lags, median, Wilcoxon p_FDR | vs acceleration: leads/lags, median, Wilcoxon p_FDR |
|---|---|---|---|
| Var(energy) | peak | 7/13, +11, 0.175 | 8/12, +2.75, 0.332 |
| Var(energy) | trough | 2/19, +36, 0.0003 | 2/19, +32, 0.0006 |
| Var(J) | peak | 4/17, +21, 0.0075 | 5/16, +18, 0.032 |
| Var(J) | trough | 6/15, +27, 0.014 | 6/15, +28, 0.024 |
| Var(h) | peak | 10/10, -1, 0.94 | 12/8, -9, 0.28 |
| Var(h) | trough | 1/19, +27, 0.0003 | 2/18, +17, 0.0006 |

Reading: the coupling-term fluctuation peak follows both kinematic peaks (about 18 to 21 samples later) and survives FDR correction; the field-term fluctuation peak is time-locked to the kinematic peak with no lead or lag. This is the clearest field-versus-coupling dissociation found so far, and it again places coupling-driven activity after peak velocity, in deceleration. Status: pilot. No written findings document yet, and the window size (W = 25, step 5) has not been varied.

## 4. Results that must be regenerated before citing

Four code bugs were fixed and deployed on the cluster (code_fixes_summary.md). The outputs of these scripts still reflect pre-fix numbers:
- `arbitration_j_many.py`: xcorr lag sign was inverted, so every earlier xcorr lead/lag claim had the opposite sign.
- `test_collective_j_timing.py`: the earlier default only included sessions with a J-peak (selection bias).
- `aggregate_transition_analysis.py`: threshold was chosen post hoc, stimulus repeats were pooled as independent, and there was no FDR correction. The cited "pooled Cohen's d = -0.114, p = 0.003" (energy spikes to deceleration) must be recomputed.
- Earlier note-level claims ("J lags acceleration peak in ~70% of sessions, mean +16.4"; P(K) collective chi-squared/Fisher p-values) came from mixed dataset snapshots and pre-fix code. Treat them as provisional.

## 5. Not supported or abandoned

- Decision tree predicting J-peaks: LOSO-CV accuracy is at or below the majority baseline in 6 of 8 configurations, so it does not generalize. The user considers it exploratory and abandoned.
- Position-kinematics effect: flagged by the authors as a likely session or window confound.

## 6. Data-quality flags

- Sessions 210620 (pair r = 0.985, J about 1.97) and 220515 (r = 0.89) likely contain a duplicated or split sorted unit. Neither has a raw .mat for cross-checking. Consider excluding them or merging the unit, then re-running.

## 7. Open next steps (priority order)

1. Regenerate the outputs of the three fixed scripts and update RESULTS_SUMMARY.md.
2. Write up the criticality pilot and test its robustness to window size.
3. Run the H(t) field-versus-coupling decomposition across all sessions.
4. Run the surrogate re-fit check to validate the Ising model specifically.
5. Re-run the timing tests with the acceleration zero-crossing as the phase boundary and a pre-registered window, with and without 210620/220515.
