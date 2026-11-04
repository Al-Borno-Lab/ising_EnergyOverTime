# E16 · 1-ms field-term peak and trough timing vs peak acceleration and peak velocity: results

> **Internal only. Not reported in the paper** (amendment A2, Gunnar 2026-11-03).
> Design: `PREREGISTRATION_2026-11-03.md` with amendments A1 and A2.

**Runs:** synthetic 5058414; real 5058415 (21/21 sessions, 63 conditions × 3 bases, all fits converged).

## Summary (plain language)
- **The median curve's pattern is not a per-condition effect.** The session-median 1-ms field term appears to peak at peak acceleration and trough at peak velocity, but condition by condition this does not hold.
- **Trough:** it falls near peak velocity no more often than in rest windows, under any basis.
- **Peak:** it falls near peak acceleration in about 3% of conditions. The synthetic null showed this test is biased against peaks, so a null or negative peak result cannot be interpreted.
- **Unchanged:** the 10-ms per-session findings (field-term dip at peak acceleration, 20/21 sessions; better timing than the mean rate) and the qualitative 1-ms observations (dip survives, coupling-term change does not).

## Results
| # | Result | Key numbers | Asset | Run |
|---|---|---|---|---|
| 1 | Synthetic: planted peak and trough recovered | Z1 (U0): peak 5/6, trough 5/6; median errors 8.5 and 9.5 ms | e16_synthetic_validation.csv | 5058414 |
| 2 | Synthetic null: peak test biased against peaks; trough test fine | Z0 (U0): D_peak −0.33 (fails ±0.2), D_trough +0.08; STD: −0.08 and +0.25 | e16_synthetic_validation.csv | 5058414 |
| 3 | **Trough at peak velocity: not supported** | U0: reach 0.21 vs rest about the same; HL 0.00 (95% CI −0.17 to +0.17), 7 positive / 8 negative sessions, p = 0.61. U25 p = 0.21; STD reach 0.22 vs rest 0.23 | e16_summary.json, e16_hit_rates.csv | 5058415 |
| 4 | Peak at peak acceleration: uninterpretable (biased test) | U0 reach hit rate 0.03 vs rest about 0.08; HL −0.08, 0/12 positive, p = 0.0015. Same under U25 and STD | e16_summary.json, e16_hit_rates.csv | 5058415 |

Verdict as registered: **not supported**.

## Caveats and open issues
- **Landmark resolution.** Landmarks are on a 10-ms grid, and the search uses a single extremum on 5-ms-smoothed 1-ms traces, which is noisy. The planted test shows the method can find a real 40-ms peak-to-trough structure when one exists.
- **What the median curve shows.** It can still be described as a group-average feature, as in the exploratory figure, but not as a per-condition timing result.

## Entries for RESULTS_LOG_FOR_PAPER.md
- D (not supported): "the 1-ms field term peaks at peak acceleration and troughs at peak velocity" holds only for the group median, not per condition (E16, internal).
