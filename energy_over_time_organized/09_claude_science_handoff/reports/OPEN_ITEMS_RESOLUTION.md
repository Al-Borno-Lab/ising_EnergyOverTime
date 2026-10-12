# Resolution of the pre-submission open items (2026-09-29)

These are the six open items from Section E of RESULTS_LOG_FOR_PAPER.md. Every fit used the project's own `fit_ising_model` (coniii MCH) inside `inverse-ising-arm-2.sif`. Group statistics are session-level (n = 21) Wilcoxon tests with BH-FDR. Time: 1 sample = 10 ms.

| # | Item | Verdict |
|---|---|---|
| 6 | Bin width | **Confirmed 10 ms** |
| 2 | Surrogate refit | **Passed** |
| 4 | Duplicated units | **Resolved; conclusions unchanged** |
| 1 | Co-silence confound | **Resolved, with a reinterpretation** |
| 3 | Time-resolved model comparison | **Resolved, with a reinterpretation** |
| 5 | Regenerate fixed scripts | **Done; three legacy claims withdrawn or replaced** |

## 0. Solver validation
Refitting 3 conditions (N = 12, 15, 24) reproduces the stored parameters: r(J) = 0.998-0.999, r(h) = 1.000, slope 0.98-1.02 (`solver_validation.csv`).

## 6. Bin width
The time column has dt = 0.010 s in all 1,088 neural and kinematic matrices of the 9 raw .mat files, consistent with `BIN_MS = 10.0`. All lags below are given in ms.

## 2. Surrogate refit
Each condition got 100 rate-preserving surrogates (each neuron's binarized train permuted independently), refitted with identical settings, plus 3 real refits.
- **Mean |J_ij|:** real exceeds all 100 surrogates in **63/63** conditions (median 4.6x the surrogate mean; min z = 3.6, median z = 53).
- **Net coupling ΣJ_ij:** real exceeds all surrogates in 62/63. The exception is 210622 stim 2, with N = 3 neurons.
- **Scope:** the permutation removes every temporal structure, including reach-locked rate modulation shared across neurons. So this shows that the couplings capture structure beyond *stationary* independent firing; it does not by itself separate coupling from shared rate modulation (see item 3).

![Surrogate refit]({{artifact:art_684bf995-0269-418e-b995-2eac882ad4a9}})

## 4. Duplicated units (210620, 220515)
- **Identification:** the top-correlated pair is the same in all three stimuli: 210620 units 1 and 8 (r = 0.98, equal rates) and 220515 units 13 and 15 (r = 0.89-0.95). The next-highest pair correlation is ≤ 0.59.
- **Refit:** one unit of each pair was dropped and the 6 conditions refitted.
- **Result:** no headline test changes (`dedup_effect.csv`):
  - coupling ΔE contribution p = 1e-5 / 1e-4 / 7e-4 (acceleration / deceleration / post);
  - field ΔE p ≥ 0.84;
  - firing rate more movement-modulated than J, p = 0.006;
  - ρ(J modulation, FR modulation) = 0.74.

## 1. Co-silence confound
**Pair-type split.** ΣJss was split per bin into co-active (+1,+1), co-silent (−1,−1) and discordant pairs; the parts sum to the stored `j` column to 2e-14.
- The co-active and co-silent contributions are individually not significant.
- They follow the firing-rate change in opposite directions across sessions (ρ = +0.89 and −0.91).
- The consistent movement effect is fewer discordant pairs among positively coupled neurons (p_FDR ≤ 0.003 in every phase).

**Rate vs covariance split.** The trial-averaged ΣJss(t) was split into the part predicted from each neuron's time-varying rate alone, Σ J_ij m_i(t) m_j(t) with the 1/R self-term removed, and the remaining across-reach covariance part.
- **Total:** the per-pair J rise is consistent (18/21 sessions, movement window p_FDR = 0.014).
- **Neither part alone** changes consistently during the reach (rate part 13/21, covariance part 11/21, all p_FDR > 0.16).
- **Share:** the covariance part accounts for a median 36% of the rise. Sessions whose firing rate falls get their J rise from the covariance part, and sessions whose rate rises get it from the rate part (ρ = −0.66 and +0.64 vs ΔFR).
- **Late rise:** the covariance part rises 500-1500 ms after peak velocity (13/21 sessions, p_FDR = 0.053).

**Interpretation.** The coupling-driven energy drop is robust, but it is not evidence of increased pairwise coordination as such. About two-thirds of it, and in some sessions all of it, reflects the population's rate trajectory moving into states the fitted couplings favour.

## 3. Time-resolved Ising vs independent model
Setup: 5-fold cross-validation over reaches, with a per-bin held-out log-likelihood ratio. log Z was computed exactly for N ≤ 22 and by thermodynamic integration above that; the two methods agree within 0.004 nats where both apply.
- **Against a constant-rate independent model:** Ising wins overall (21/21 sessions, 61/63 conditions; median 0.21 nats/bin). Its advantage grows during movement (acceleration +0.41 nats, 18/21; movement window 20/21; p_FDR < 1e-4).
- **Against a time-varying-rate independent model** (per-bin rates learned from the training reaches, 11-bin smoothing):
  - Ising is better at **baseline** (16/21, p_FDR = 0.004);
  - it is **worse during acceleration** (2/21 positive, p_FDR = 0.004);
  - it is **not different** during deceleration or post.
- **Conclusion:** the movement-time "Ising advantage" is reach-locked rate modulation, which a stationary Ising model can only absorb through its couplings. It does not show extra pairwise coupling during the reach. Pairwise structure beyond rates is detectable at rest.
- **Late window:** 500-1500 ms after peak velocity, Ising is better than the time-varying independent model on average, but only in 12/21 sessions (p = 0.07). This matches the late covariance rise in item 1.
- **Flat-firing-rate conditions:** these include 210511 stim 1. They show no movement-time advantage over the time-varying independent model (2/21 positive).

![Coupling source]({{artifact:art_f79ff9ab-6cc1-4959-b3b8-21436f0032a4}})

## 5. Regenerated scripts
All outputs are in `07_presubmission_checks/regenerated/` on the cluster (`regenerated_vs_legacy.csv`).
- **`arbitration_j_many.py`, window 390-410:** the J jump lags acceleration in 40/63 conditions, which replaces the legacy "~70%". The same search in pre-reach control windows gives 75% "lag". Paired real minus control is +0.1 samples, p = 0.79. **Withdrawn.**
- **`aggregate_transition_analysis.py`:** fixed threshold 2.0, reps collapsed per session, FDR applied.
  - Acceleration: stim 0 gives d = −0.103 [−0.178, −0.028], p_FDR = 0.021. Stim 1 gives d = −0.004 (p = 0.92) and stim 2 gives +0.031 (p = 0.64).
  - The legacy d = −0.114, p = 0.003 is **replaced**: the effect is small and does not replicate across stimuli.
- **`test_collective_j_timing.py`, unbiased primary analysis:** median distance to peak acceleration is J 12, h 13, FR 12 and energy 12 bins, with no pairwise difference (all p ≥ 0.41). In the J-gated secondary analysis, firing rate is closer than J (p = 0.042). The claim that J is the best indicator of peak acceleration is **withdrawn**.

### Code issues found while regenerating (project code not modified)
- `aggregate_transition_analysis.py` labels sessions by `sdir.parent.name`. In the organized layout this is `full_reach` for every session, so all 21 sessions overwrite one another and pool as a single animal. It was run here on a per-session copy tree.
- `pk_j_analysis.py`, line 449, uses a backslash inside an f-string. That needs Python ≥ 3.12 and fails in the project container (Python 3.10). It was run here in a separate Python 3.12 environment.
- Both timing scripts import `src.*`, which needs `shared/` on PYTHONPATH.

## Corrected peak-timing figure
This figure supersedes panel b of `phase_timing_summary.png` and the peak-timing rows of `sensitivity_robustness.png`. Each signal's maximum is compared with the same search run in pre-reach control windows.
- **Energy maxima** occur earlier than control (−350 ms, p_FDR = 0.017), but this holds in only 2 of 6 window variants.
- **J, h and firing rate** do not differ from control in any variant.

![Peak timing vs control]({{artifact:art_933e4afc-1d96-4261-95e2-0d910cc464a2}})

## Remaining open question
Is there pairwise coupling beyond shared rate modulation during the reach? A stationary Ising model cannot answer this. The next test is a time-varying-field model: h_i(t) learned from the training reaches with J fixed, compared against the time-varying independent model.
