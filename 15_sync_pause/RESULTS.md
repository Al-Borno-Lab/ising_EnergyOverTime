# E15 · Synchronous pauses: shared drive vs fixed coupling: results

> Design: `PREREGISTRATION_2026-10-27.md` with amendments A1 and A2, both made before any real result was read.

**Runs** (`runs/RUN_LOG.md`):
- library check 4588232: PASS;
- synthetic validation 4588233: PASS;
- real data 4588234, plus 4588294 for 210512 and 210608.

**Outputs:**
- raw: `~/ising_wt/e15/15_sync_pause/results/`;
- figure: `assets/figures/e15_pause_timecourse.png`;
- tables: `assets/tables/e15_summary.json`, `e15_tests.csv`, `e15_conditions.csv`, `e15_calibration.csv`, `e15_synthetic_validation.csv`, `e15_posthoc_W_minus_C.csv`.

## Summary (plain language)
- **At rest:** groups of Purkinje cells fall silent together about 20% more often than their firing rates alone predict. This holds in every session, and the fixed-coupling Ising model reproduces it.
- **During the reach:** there is no consistent extra population pause around peak acceleration. The field-term dip is a polarization of the population (pausers quiet, bursters active), not a joint silence.
- **The key question is unresolved.** We asked whether fixed couplings correctly predict how pause synchrony changes during the reach. The pre-registered test is **inconclusive**: the answer depends on the reference window, and the 2022 cohort carries the effect.

## Results
| # | Result | Key numbers | Asset | Run |
|---|---|---|---|---|
| 1 | Pre-registered verdict: inconclusive | A_obs0 −0.087 (95% CI −0.19…−0.02, 6/21 > 0, q = 0.048); A_10 −0.134 (−0.32…−0.03, 9/21, q = 0.048); A_obs1 +0.036 (−0.01…+0.14, 12/21, q = 0.12, equivalent within ±log 1.25) | e15_summary.json, e15_tests.csv | 4588234/4588294 |
| 2 | Pauses exceed the rates-only prediction at rest | L_obs0 +0.19 (0.14–0.25), 21/21, p = 1e-6 | e15_tests.csv, figure panel a | same |
| 3 | The fixed-J model accounts for the excess at rest (slight over-prediction) | L_obs1 −0.08 (−0.25…−0.005), 6/21 > 0, p = 0.03 | e15_tests.csv | same |
| 4 | No consistent rise in pause rate in the reach window | R_obs +0.195 (−0.16…+0.54), 12/21, p = 0.30; post window 11/21 | e15_tests.csv | same |
| 5 | Bursts behave like pauses: excess over rates, tracked by M1 | figure panels d–f; A_obs1 (burst, W) +0.105, p = 0.026, secondary and uncorrected | e15_tests.csv | same |
| 6 | Method check passes | S0 median abs(A) ≤ 0.06; S1 median abs(A_obs1) 0.05 (fixed J explains a −5.4 change in 220516_0); S2 detected 5/6 | e15_synthetic_validation.csv | 4588233 |

Why the verdict is inconclusive: the pattern matches rule (b) with the opposite sign. The coupling-driven excess of pauses shrinks during the reach, fixed J predicts that shrinkage, and nothing is left over. Rule (b) was registered for a rise, and the A2 sign flip applies only when R_obs < 0. Here R_obs is positive but not significant.

## Caveats and open issues
- **Rest-control drift.** Two rest windows already differ: A_obs0 −0.054 (3/21 > 0, p < 0.001), A_obs1 −0.064 (p = 0.03). That drift is about 60% of the reach effect on A_obs0. The fixed-J *prediction* does not drift at rest (A_10 +0.01), so its predicted drop is reach-specific.
- **POST HOC** (`e15_posthoc_W_minus_C.csv`, planning only). Measured against the rest control:
  - the data's excess over rates does not change during the reach (−0.003, 9/21, p = 0.37);
  - fixed J predicts a drop (−0.15, p = 0.03);
  - so the data keep more synchrony than fixed J predicts (+0.08, 15/21, p = 0.007; 2021 alone p = 0.02).

  The choice of reference decides between "fixed J explains it" and "more synchrony than fixed J predicts".
- **Cohort.** 2021 is null on the primary statistics (A_obs1 −0.001, p = 0.98). 2022 carries the effects: A_10 −0.53, A_obs1 +0.23 (6/6, p = 0.03). The 2022 sessions are also where pseudo-likelihood couplings are least reliable.
- **Calibration.** M1 slightly under-predicts mean activity in the reach window (9.60 vs 9.98 active cells per bin), which biases it toward more predicted pauses. The predicted drop happens despite this.
- **Review.** Independent code review is done. The spot-check and the reproduction are pending (`REVIEW_2026-10-27.md`).

**Suggested follow-ups** (need approval):
1. Pre-register a test that uses several rest windows as the drift null.
2. Re-estimate J for the 2022 sessions with exact or MCH fitting and re-run A_10 and A_obs1.
3. Link to E14 N3 at 1–5 ms.

## Entries for RESULTS_LOG_FOR_PAPER.md (proposed; add after review)
- New B-type: synchronous population pauses exceed the rates-only prediction at rest in 21/21 sessions (about +20%), and the fixed-J model accounts for them.
- New B-type: no consistent rise in population pause rate around peak acceleration (12/21).
- New C-type (open): whether fixed J predicts the reach-time change in pause synchrony is unresolved.
