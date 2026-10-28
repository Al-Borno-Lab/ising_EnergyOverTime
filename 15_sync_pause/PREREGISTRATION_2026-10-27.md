# E15 pre-registration (2026-10-27, frozen before any E15 code touched real data)

Approved by Gunnar on 2026-10-27 ("lets do it, I approve the experiment").

## Background (what we already know; not tested here)
- Couplings J add held-out prediction beyond time-varying rates at all times, including rest (A7; S6: 21/21 sessions). They do not detectably change during the reach (A8).
- The field term −Σh·s dips at peak acceleration. The dip is a population-polarization effect (E12), comes from each cell's reach-locked time course (S2: preserved by the trial shuffle, destroyed by time shifts), and does not need the coupled model.
- Open question: when the population pauses together during the reach, is that **(a)** only shared reach-locked drive (rates), **(b)** shared drive amplified by the *fixed* couplings, or **(c)** something the fixed-J pairwise model cannot produce?

## Events
For each bin, K = number of recorded cells that spike in that 10-ms bin.
- **Pause** (primary): K ≤ k_lo, where k_lo = `np.quantile(K_train_baseline, 0.10, method="lower")`. K_train_baseline pools all bins of the baseline window over the *training* reaches of the current fold.
- **Burst** (secondary): K ≥ k_hi = `np.quantile(K_train_baseline, 0.90, method="higher")`.
- **Sensitivity:** the same events at the 0.05 / 0.95 quantiles.

Thresholds are fixed per fold from training data only, and the same threshold is applied to the held-out data and to both models.

## Models (from `09_claude_science_handoff/code/tvfield_model.py`, unchanged)
- **M0:** time-varying fields, J = 0. P(K | t) is Poisson-binomial with p_i(t) = sigmoid(2 h_i(t)).
- **M1:** time-varying fields + fixed symmetric J. P(K | t) is computed exactly by enumerating 2^N states for N ≤ 20, and by Gibbs sampling for N > 20 (400 chains per bin, 250 sweeps, first 50 discarded; validated against exact enumeration on N ≤ 20 conditions before use).
- **Fitting:** basis `bspline_basis(zero_cross)` (67 cubic B-splines); pseudo-likelihood with lamB = 1e-5, lamJ = 1e-3; 5-fold CV over reaches using the stored `fold_k`. Exclusions: reaches with `exclude_k == 1` are dropped.

For each held-out reach r and bin t:
- observed event indicator: o(r, t);
- model event probabilities: q0(r, t) and q1(r, t), from the fold that did not train on r.

e_obs(t), e0(t) and e1(t) are the means over all reaches.

## Windows (bins relative to the condition's `zero_cross`, i.e. peak velocity; 10 ms per bin)
| name | bins | ms |
|---|---|---|
| baseline B | zc−100 … zc−41 | −1000 … −410 |
| reach W (primary) | zc−30 … zc+10 | −300 … +100 |
| post P (secondary) | zc+11 … zc+50 | +110 … +500 |
| rest control C | zc−200 … zc−160 | −2000 … −1600 |

The code asserts these values and writes them into every output row.

## Statistics per condition
Each window rate is the mean over the window's bins and all reaches. For any window X, ε = 0.5 / (number of reach-bins in the window), and every rate below is taken with ε added (e.g. e_obs,X stands for e_obs,X + ε).

| quantity | formula | meaning |
|---|---|---|
| R_obs | log(e_obs,W / e_obs,B) | pause rate rises during the reach (descriptive) |
| A_obs0 | log(e_obs/e0)_W − log(e_obs/e0)_B | **data excess over rates-only grows during the reach** |
| A_10 | log(e1/e0)_W − log(e1/e0)_B | **fixed couplings predict an amplification** |
| A_obs1 | log(e_obs/e1)_W − log(e_obs/e1)_B | **residual beyond the fixed-J model** |
| L_obs0 | log(e_obs/e0)_B | excess over rates at rest (level) |
| L_obs1 | log(e_obs/e1)_B | excess over M1 at rest (level) |

**Calibration:** observed vs predicted mean K per window, for both models.

## Unit, tests, decision rule
- **Unit:** the session (mean of its 3 conditions); n = 21.
- **Tests:** two-sided Wilcoxon signed-rank vs 0; Hodges–Lehmann estimate with 95% CI (bootstrap over sessions, 10,000 resamples, seed 15).
- **Primary family** (BH-FDR, q = 0.05): A_obs0, A_10, A_obs1 for **pause** in window W.
- **Equivalence for A_obs1:** the fixed-J model "accounts for" the reach change if the 90% CI of A_obs1 lies within ±log(1.25) = ±0.223.

| Outcome | Rule |
|---|---|
| (a) shared drive only | R_obs > 0, A_obs0 not significant **and** its 90% CI within ±0.223 |
| (b) fixed coupling amplifies shared drive | A_obs0 > 0 (FDR-significant), A_10 > 0 (FDR-significant), A_obs1 equivalent to 0 |
| (c) beyond the fixed pairwise model | A_obs1 ≠ 0 (FDR-significant) |
| inconclusive | anything else; report CIs |

If R_obs is not positive (pauses do not rise in W), report that first. The question is then about pause *suppression*, and the same rules apply with signs read accordingly.

## Secondary / sensitivity (not in the FDR family; reported with CIs)
- Burst events.
- Quantile 0.05 / 0.95.
- Post window P.
- Rest-control window C in place of W: all A statistics are expected ≈ 0, as a timing control.
- Cohort 2021 vs 2022.
- Sessions with N ≥ 8 in all blocks.
- Exact-only conditions (N ≤ 20).

## Synthetic validation (method check; must pass before real data is analysed)
Ground truth is taken from real conditions: the M1 fit on all reaches (h(t), J) or the M0 fit, with real R, folds, zero_cross and N. The six conditions include at least one with N > 20 (Gibbs path). Data are sampled from the true model per bin: exact for N ≤ 20, long-burn Gibbs above.

| Scenario | Ground truth | Pass criterion |
|---|---|---|
| S0 | J = 0 (M0 truth) | median |A_obs0|, |A_10|, |A_obs1| < 0.1 |
| S1 | fixed real J | median |A_obs1| < 0.1 |
| S2 | J × 2 inside window W only | A_obs1 > 0 in ≥ 5/6 conditions |

Results are labelled synthetic and never pooled with real data.

## Review
`code/stats_e15.py` is written and reviewed by an independent agent before real results are read, as PROTOCOL.md requires.

## Amendment A1 (2026-10-27, before any result was read)
After exclusions, two conditions (210512 stim 2, 9 reaches; 210608 stim 1, 10 reaches) have no reaches in one of the five stored folds. The first submission (array 4588234) stopped on the code's fold assertion for those two sessions. No statistic was looked at. Change: cross-validation loops over the folds present after exclusions, using the stored `fold_k`, so every reach is still held out exactly once; the number of folds is written into each row (`n_folds`). Conditions with all five folds are computed exactly as before. Only the two affected session tasks (210512, 210608) are rerun.
