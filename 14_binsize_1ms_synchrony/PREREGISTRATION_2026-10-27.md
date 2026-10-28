# E14 pre-registration (2026-10-27, frozen before any E14 analysis code touched the data)

Approved by Gunnar on 2026-10-27 ("yep lets do e14"). The owner changes from grad-researcher-2 (session ended) to the orchestrator.

## What has already been seen (disclosure)
- **E14 extraction and data gates** (job 4363528, read today): all 21 sessions pass G1–G3. `c1` holds 1-ms counts that re-bin exactly to the 10-ms data.
- **Exploratory 1-ms refit** (evidence package S3, separate code), already seen:
  - 1-ms M1 energy terms: the field-term dip survives about 45 ms later; the coupling-term dip is absent;
  - J at 1 ms vs J at 10 ms: r = 0.61;
  - 210511 unit 11 looks duplicated.
- **Never computed before this document:**
  - no synchrony index (SI);
  - no epoch coupling scale κ at any bin size;
  - no model-predicted SI.
- A8 (10 ms): no evidence of coupling change during the reach. The acceleration-epoch CI is −28% to +93%, leaning positive.

## Question
Nashef et al. (same recordings; 1-ms jPETH/CCG) report Purkinje-cell co-firing rising during the reach, mostly in reach-suppressed cells and about 100 ms before endpoint. At 10 ms our model shows no coupling change. Three explanations are possible:
- **(a) timescale:** the change lives at 1–5 ms;
- **(b) operating point:** fixed J with moving fields produces the measured synchrony change;
- **(c) cell selection:** the change exists only in rate-suppressed pairs.

## Data
- `results/spk/<session>.npz` (`c1` (R, 7000, N), 1-ms counts, rows 100:800), plus `stim`, `exclude`, `idx_cross` and `idx_end`. The last two are 10-ms bins within the 700-bin window.
- Reaches with `exclude == 1` are dropped.
- Spikes are binarized per bin (count > 0).
- **Primary block: stim 0** (baseline, no laser), matching Nashef et al., who did not analyse stimulation blocks. Stim 1 and 2 enter N1 as a secondary analysis only.

## Windows (per reach; ms converted to bins of the current bin size)
| name | definition |
|---|---|
| B (baseline) | 1-cm crossing −1000 … −400 ms (crossing = `idx_cross`×10 ms) |
| E (endpoint, primary) | `idx_end`×10 ms −100 … 0 ms (the 100 ms before each reach's endpoint) |
| A (acceleration, secondary) | crossing −90 … 0 ms |

Reaches with a non-finite `idx_cross` or `idx_end`, or a window outside 0 … 7000 ms, are dropped from window statistics; they are still used for fitting.

## Cell classes
- A cell is **suppressed** in a block if its mean 1-ms spike probability in E is lower than in B, over that block's reaches.
- **Suppressed pairs** = both cells suppressed. **Other pairs** = all remaining pairs.
- This approximates Nashef's cluster 1; it is not their clustering.

## N1: synchrony index (model-free)
For a pair (i, j), window X and 1-ms binary states:
- SI = P(both spike in the same bin) / (P(i spikes) · P(j spikes)), pooled over the reaches and window bins.
- **Shuffle correction:** the same SI with cell j taken from a different reach (20 random derangements per pair, seed 14), keeping window-relative time aligned. ΔSI_X = SI_obs − mean SI_shuffle.
- **Statistic:** D_pair = ΔSI_E − ΔSI_B. Pairs where either cell has no spike in a window are skipped.
- **Summary:** per condition, the mean over pairs in each class; per session, the stim 0 value.
- **N1 (primary):** D for suppressed pairs > 0. Two-sided Wilcoxon over sessions that have at least one suppressed pair.
- **Secondary:** other pairs; all pairs; window A; all blocks.

## N2: epoch coupling scale at 1, 5, 10, 20 ms
- Fields use the same B-spline construction as `tvfield_model.bspline_basis` with knots fixed in ms: fine every 50 ms from −600 to +1600 ms around the crossing, coarse every 250 ms elsewhere, so 67 basis functions at every bin size.
- M1 is fitted by pseudo-likelihood (lamB = 1e-5, lamJ = 1e-3) on all stim 0 reaches.
- Then, with J fixed, two scales and the fields are fitted jointly by pseudo-likelihood:
  - κ_s for suppressed pairs, κ_o for other pairs;
  - J_eff,ij(r, t) = (1 + κ_g · 1[(r, t) ∈ E]) J_ij.
- Penalty: none on κ.
- Bin size b: a bin is "in E" if it lies wholly inside the reach's E window.
- **N2 (primary):** κ_s at 1 ms ≠ 0.
- **Secondary:** κ_o and κ at 5, 10 and 20 ms; the timescale contrast κ_s(1 ms) − κ_s(10 ms); window A in place of E.
- **Rationale for pseudo-likelihood instead of E11's exact κ:** exact likelihood at 7000 bins per reach costs about 10× E11. The PL κ estimator must pass the synthetic test below before it is used.

## N3: operating point (does fixed J predict the SI change?)
- From the 1-ms M1 fit (κ = 0), simulate R reaches by Gibbs sampling: 200 burn-in sweeps; each (reach, bin) is an independent draw from the model at that bin.
- Compute D for suppressed pairs exactly as in N1, with the same reaches' windows and cell classes taken from the real data. Average over 5 simulated replicates.
- **N3 statistic:** D_obs − D_pred (suppressed pairs, stim 0, 1 ms).
- **N3:** fixed J accounts for the change if the 90% bootstrap CI of the Hodges–Lehmann estimate lies within ±50% of the median |D_obs|. Equivalence margin m = 0.5 · median_sessions |D_obs|.
- **Descriptive:** fraction explained = D_pred / D_obs per session (median reported).

## Statistics
- **Unit:** session. Primary is stim 0.
- **Tests:** two-sided Wilcoxon signed-rank; Hodges–Lehmann with 10,000 bootstrap resamples (seed 14), 95% and 90% CIs.
- **Primary family** {N1, N2, N3-difference}: BH-FDR at q = 0.05.
- **Also reported:** cohort (2021 vs 2022) and a sensitivity analysis without 210511 (duplicated-unit flag).

## Decision rule
| Outcome | Rule |
|---|---|
| N1 not significant | Nashef's rise is not reproduced with this window and class proxy. Report it; check windows and cells against their paper. |
| N1 significant, D > 0 | Then: |
| — (b) operating point | N3 equivalent: fixed J with moving fields reproduces the synchrony rise. The papers agree and our model gives the mechanism. |
| — (a) timescale | N3 not equivalent and κ_s(1 ms) > 0 (FDR-significant), with κ_s(10 ms) not significant: coupling changes at fine timescales only. Revise A8 wording. |
| — mixed or other | Report the CIs. |
| (c) selection | Reported descriptively: suppressed vs other pairs for N1 and κ. |

## Synthetic validation (must pass before real-data results are read)
- **Ground truth:** 1-ms M1 fits of four real stim 0 conditions (N from 8 to 25, including one with N > 20), with real R and real windows. Simulated by Gibbs, 200 burn-in sweeps.
- **Pass rule:** the estimator recovers planted κ_s within ±0.25, and the N3 residual follows the scenario's expected sign in at least 3/4 conditions.

| Scenario | Ground truth | Pass criterion |
|---|---|---|
| T0 | κ = 0 | median |κ_s| < 0.15 at 1 ms; N3 residual has median |·| < m |
| T1 | κ_s = +0.5 in E | κ_s recovered: median within ±0.25 of 0.5, positive in ≥ 3/4; N3 residual > 0 in ≥ 3/4 |
| T2 | κ = 0, plus a shared per-reach gain on all fields in E (log-normal, SD 0.3) | κ false-positive check: report κ_s and N1. Not a pass/fail item, but it decides whether a positive κ in real data can be read as coupling. |

## Review
`code/stats_e14.py` and the estimator code are reviewed by an independent agent before real results are read (PROTOCOL.md).

## Clarifications (2026-10-27, written while building the code, before any E14 analysis touched the data)
- **T2 "gain":** each reach r gets a field offset δ_r ~ N(0, 0.3), added to every cell's field inside E. This multiplies each cell's firing odds by exp(2δ_r) in that reach, i.e. a log-normal, shared, trial-to-trial gain.
- **Basis centre:** the 1-cm crossing, fixed at 4000 ms of the 7000-ms window (bin 400 at 10 ms = row 500 of the raw reach segment). This replaces the per-condition `zero_cross` used in E10 to E15, which lies within 2 to 8 bins of 4000 ms in every condition (392 to 403). With 50-ms knots everywhere in −600 … +1600 ms, the difference is immaterial.
- **Reaches in the SI analysis:** only reaches with valid B and E windows; this gives fixed window lengths at 1 ms (600 and 100 bins).

## Amendment A1 (2026-10-27, after independent code review, before any real-data result was read)
The review found the estimator correct:
- the κ gradient matches finite differences;
- the PL loss equals a brute-force J_eff loss;
- the SI, basis and window rules check out.

Changes and clarifications:
1. **Synthetic T0 pass rule:** fixed to the registered "median of |κ_s|" and "median of |residual|". The code had taken |median|.
2. **SI shuffle:** each of the 20 draws applies one random derangement of reaches to all pairs at once, not a separate derangement per pair. The two are statistically equivalent.
3. **Reaches whose `idx_end` is the last bin (699):** three such reaches exist in stim 0 (210514, 220518, 220520; 220520's B window is invalid). They are **kept**, as the window rule specifies. Their E window lies at 6890–6990 ms, and the true endpoint may be later. This is disclosed; it affects at most two reaches out of about 380.
4. **Window A classes:** suppressed classes are computed from the E-window analysis and reused for A. The code compares B from the A-valid reaches with E from the E-valid reaches; the two reach sets are identical in these data, so outputs are unchanged. The running code is left as is, for reproducibility.
5. **Timescale rule (a):** "κ_s(10 ms) not significant" uses an unadjusted Wilcoxon p ≥ 0.05.
6. **Selection (c):** paired rows added, D_supp − D_other and κ_s − κ_o. These are descriptive.
7. **Not changed:**
   - **Thread oversubscription in sbatch:** a runtime issue only.
   - **`class_agreement` in the synthetic table includes the diagonal:** descriptive only.
   - **Synthetic margin m taken from T0's observed D:** a stand-in for the real margin.
8. **Python start method:** `fit_M1` passes data to workers through a module global, which requires fork. The cluster container is checked before results are used; this was verified to work in the S3 1-ms runs.
