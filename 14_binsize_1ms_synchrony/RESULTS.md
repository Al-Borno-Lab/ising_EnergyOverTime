# E14 · Bin-size sweep and 1-ms synchrony (Nashef reconciliation): results

> Design: `PREREGISTRATION_2026-10-27.md`, with its clarifications and amendments A1 and A2.

**Runs** (`runs/RUN_LOG.md`):
- extraction and gates: 4363528;
- synthetic validation: 4590157;
- extended null test: 4590182;
- real data: 4590158.

**Outputs:**
- figure: `assets/figures/e14_nashef_reconciliation.png`;
- tables: `assets/tables/e14_summary.json`, `e14_tests.csv`, `e14_rows.csv`, `e14_synthetic_validation.csv`, `e14_null_validation.csv`.

## Summary (plain language)
- **The model-based tests could not be validated.** The coupling-change estimate reads high even when nothing changes (+0.19 on average with no change planted), so its positive value in the real data (+0.25) cannot be interpreted. The fixed-coupling prediction test had too many false positives (12%). Both are reported as unvalidated, as decided before the real data were read (amendment A2).
- **Nashef's 1-ms synchrony rise was not reproduced** in our pre-registered test: rate-suppressed cell pairs, 100 ms before endpoint, baseline block. The change leans positive but is not significant (13/21 sessions).
- **The secondary results lean the same way.** Suppressed pairs are positive in every block, and nominally significant in the washout block (uncorrected).
- **Method finding (synthetic):** shared trial-to-trial gain fluctuations, with no coupling change at all, inflate the shuffle-corrected synchrony index strongly.

## Results
| # | Result | Key numbers | Asset | Run |
|---|---|---|---|---|
| 1 | Data gates | G1–G3 pass 21/21 (1-ms spike times re-bin exactly to the 10-ms data) | `results/spk/gates_*.csv` | 4363528 |
| 2 | **N1 (primary, pre-registered): not significant** | ΔSI change, suppressed pairs, stim 0, 100 ms before endpoint: HL +0.062 (95% CI −0.055…+0.274), 13/21 > 0, p = 0.29, q = 0.43 | e14_tests.csv, figure panel a | 4590158 |
| 3 | N1 secondary: positive lean in suppressed pairs in every block (uncorrected) | stim 1 +0.14 (p = 0.07); stim 2 +0.18 (16/19, p = 0.014); other pairs about 0; suppressed − other in stim 0 not different (p = 0.52); 2021 +0.02 (p = 0.85), 2022 +0.33 (p = 0.22); without 210511 similar | e14_tests.csv | 4590158 |
| 4 | κ estimator recovers a planted change but is biased upward under the null | T1: planted 0.5, recovered 0.37/0.47/0.47. T0x (17 conditions × 3 reps): mean +0.19, SD 0.34; bias largest where there are few suppressed pairs (up to +1.0 with 2–3 pairs); session-level false positives about 100% | e14_synthetic_validation.csv, e14_null_validation.csv | 4590157, 4590182 |
| 5 | **N2: unvalidated**; the real value is within the null bias | κ_s(1 ms) HL +0.25 (20/21); κ_s(10 ms) p = 0.18; κ_o about 0 at every bin size | e14_tests.csv, figure panel b (marked) | 4590158 |
| 6 | **N3: unvalidated** | residual observed − predicted +0.03 (CI −0.08…+0.19), not equivalent; median fraction explained 0.14 | e14_summary.json, figure panel c (marked) | 4590158 |
| 7 | Shared gain inflates SI (synthetic T2) | with no coupling change, a per-reach field offset (SD 0.3) gives ΔSI 1.2–2.7 vs 0.2–0.4 without it; κ_s barely moves (0.08–0.12) | e14_synthetic_rows.csv | 4590157 |

## Caveats and open issues
- **Proxy design.** The window and cell class approximate Nashef's (100 ms before endpoint; suppressed = lower rate in that window than at baseline). It is not their clustering or their exact test, and only the baseline block is primary.
- **The κ bias.** It probably comes from choosing the suppressed class from the same data the estimate is made on, and from very few suppressed pairs in some sessions. A held-out class definition could fix it; that needs a new pre-registration.
- **Low power.** Primary n = 21 sessions, with 12–28 baseline reaches each.

## Entries for RESULTS_LOG_FOR_PAPER.md (proposed)
- C-type (open): a 1-ms synchrony increase before endpoint in rate-suppressed pairs leans positive but is not significant in the pre-registered baseline-block test. It is nominally positive in the stimulation and washout blocks.
- Method note for the Discussion: shared trial-to-trial gain inflates shuffle-corrected synchrony indices (synthetic). The Nashef-type increase and our fixed-J result therefore need not conflict, but E14 does not resolve them.
- A8 wording unchanged: "no evidence at 10 ms; moderate changes not excluded".
