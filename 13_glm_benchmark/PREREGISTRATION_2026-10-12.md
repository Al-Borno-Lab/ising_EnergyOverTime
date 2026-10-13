# E13 pre-registration (dated 2026-10-12, author: grad-researcher-3)

Frozen BEFORE any GLM is fitted to the real spikes. Later changes appear only as dated amendments at the bottom.
**Deviation from the README:** the README says to run as an sbatch job. This session cannot submit sbatch, and the compute is light (logistic regressions), so E13 runs **locally** (laptop, multiprocessing). If it turns out to need the cluster, stop and ask.

## 0. What was already known (disclosure)
- From E07/E09 (not recomputed for this registration): M1 over M0 held-out pseudo-likelihood gain, median +0.26 nats/bin above a trial-shuffle null, 21/21 sessions (A7); no change in coupling gain between baseline and acceleration/deceleration/post (p >= 0.45; acceleration change -15%, CI -28% to +93%, A8); epoch-specific couplings do not improve held-out prediction (9/21, p = 0.71); a 200-ms-lagged control (Mlag) recovered about a quarter of the gain (0.05 vs 0.18 nats/bin).
- No GLM variant of this registration has been fitted to the real data. In E12 (same session) the author fitted M0/M1 to the real spikes for field terms but did not look at held-out likelihoods.
- Synthetic validation (section 6) runs before real data.

## 1. Data
`spikes_v2.npz`, 63 conditions (21 sessions x 3 stim blocks), reaches with `exclude_k == 0`, 10-ms bins, rows 100:800, the same 5 outer folds `fold_k` as E07. Landmarks from `09_.../assets/tables/phase_landmarks.csv`; `zc` = zero_cross. Cohort = 2021 (15 sessions) vs 2022 (6); mouse map as in E12's pre-registration (6 mice). Phases (same as 09 `phase_core.py`): baseline = [zc-100, zc-40), acceleration = [onset, zc), deceleration = [zc, decel_end), post = [decel_end, decel_end+50).

## 2. Models (per neuron i, conditional logistic regression, spins s in {-1,+1})
All share the B-spline baseline `bspline_basis(zc)` (vendored `tvfield_model.py`, penalty 1e-5 on baseline coefficients) and penalty lamJ = 1e-3 on every coupling and history weight (identical to E07). Fixed penalties (no nested CV) so that models are comparable with the stored E07 results; lamJ in {1e-4, 1e-2} is a sensitivity run (without nulls). The README's "chosen by nested CV" is replaced by this, as a stated deviation.
| id | features besides baseline |
|---|---|
| M0 | none (= GLM-0, time-varying independent) |
| M1 | the E07 model (`tv.fit(..., "M1")`: symmetrised J, fields refit); the Ising reference |
| SYNC | other cells' same-bin spins s_j(t), free asymmetric weights |
| LAG1 | other cells' previous-bin spins s_j(t-1) (zero at t = 0) |
| LAG1H | LAG1 + own history s_i(t-1), s_i(t-2), s_i(t-3) (30 ms) |
| SYNC_LAG1 | SYNC + LAG1 |
| SYNC_H | SYNC + own history |
| LAG2 | other cells' s_j(t-2) only (descriptive) |
| MLAG20 | the E07 200-ms-lag control (`tv.fit(..., "Mlag")`; reference for the "about a quarter" statement) |

## 3. Metric
Held-out conditional (pseudo-)log-likelihood summed over neurons per bin (nats/bin, as E07), 5-fold CV over reaches with `fold_k`. Per condition: mean over held-out reaches and bins (all bins, and by phase). Session value = mean over its 3 conditions. Gain of model m = LL_m - LL_M0. **Trial-shuffle null:** 20 shuffles per condition (reach identity permuted independently per neuron, as A7), refitting M0, SYNC and LAG1H on each shuffle with the same folds; **net gain = gain - mean null gain** (for the same model, same bins/phase). Nulls exist only for SYNC and LAG1H (and so for the G2/G3 quantities); other models are reported as raw gains and are labelled so.

## 4. Hypotheses and tests (unit = session, n = 21; two-sided Wilcoxon signed-rank; BH-FDR over the family below)
- **G1 (implementation check; not in the family):** per session, LL_SYNC - LL_M1 (all bins). Equivalence margin +/-0.01 nats/bin, declared when the 95% bootstrap CI of the session mean lies inside the margin. Also report net-gain SYNC vs M1 raw gain correlation across sessions.
- **G2:** ratio r_s = net gain (LAG1H) / net gain (SYNC), all bins. "Lagged coupling recovers less than half" if the 95% bootstrap CI of median r_s lies below 0.5 (session-level Wilcoxon of r_s - 0.5 is the test in the family). Secondary: same for LAG1 (no history), LAG2, MLAG20 (raw gains, descriptive), and the pooled ratio of session means.
- **G3a (A8 replication with a GLM):** for SYNC, change_p = net gain in phase p minus net gain in baseline (nats/bin), p in {acceleration, deceleration, post}; Wilcoxon vs 0 and bootstrap CI; relative change = (mean net_p / mean net_baseline) - 1 with bootstrap CI.
- **G3b:** epoch-specific vs shared coupling weights in SYNC, same design as `run_epochJ.py` (reach = [zc-50, zc+50), late = [zc+50, zc+150); held-out conditional LL of epoch model minus shared model, all bins), Wilcoxon vs 0 and CI.
- Family (BH-FDR): {G2, G3a-acceleration, G3a-deceleration, G3a-post, G3b} = 5 tests. Report cohorts separately and mouse as a random intercept (`diff ~ 1 + (1|mouse)`, MixedLM), as in E12. Decision language: "no change detected" only with the CI shown; "recovers < half" only if the CI is below 0.5.
- **G4 (descriptive, no tests):** for each condition and each of M0, M1, SYNC, LAG1H, sample population states (M1, SYNC: Gibbs sampling from the fitted conditionals, 3000 parallel chains, 40 sweeps, per bin; M0: independent Bernoulli; LAG1H: forward simulation in time from a zero start). Compare P(K) (K = number of spikes in a bin) with the data's, per phase window (baseline and the +/-500 ms reach window), by total-variation distance; report session medians and counts of sessions where the Ising model is closer. Note: the fitted GLMs are conditionals, not joint distributions, so their "samples" are pseudo-Gibbs / forward simulations; this is a stated limitation.
- Sensitivity (not in the family): all reaches (no exclusion); sessions with N >= 5; lamJ = 1e-4 and 1e-2.

## 5. Interpretation rules
- G1 holds -> the GLM-sync benchmark is the same model as M1; A7's existence claim then does not require Ising language (it is method-independent).
- G2 as stated -> the A7 gain is mostly same-bin: a lagged-only GLM cannot reproduce it; with the 10-ms bin this is consistent with shared drive or synchrony within 10 ms and does not say which.
- G3 flat for SYNC -> A8 is method-independent. If SYNC shows a change and M1 did not, report the discrepancy.
- Report ties and failures as such. No claim that the Ising model beats a GLM on held-out prediction; what it adds is a normalised joint distribution, P(K) and thermodynamics (G4, and E10).

## 6. Method validation (synthetic, labelled as such)
(i) Simulated exact Ising with same-bin J and time-varying fields: SYNC LL equals M1 LL within the G1 margin; LAG1 and LAG1H find no net gain (J acts only within a bin; bins independent in time). (ii) A kinetic positive control (spike at t depends on spins at t-1): LAG1 finds the gain, SYNC does not. (iii) J constant: G3a changes ~ 0; J doubled in the acceleration window: G3a detects it (power check; the acceleration phase is only ~10 bins).

## Amendments
- **A1 (2026-10-12, after the synthetic validation, before any real-data fit).**
  1. *What the validation showed.* G1 holds (SYNC = M1 within 1e-5 nats/bin); LAG1/LAG1H net gain is ~0 when J acts only within a bin; a kinetic positive control is detected by LAG1. But with CONSTANT J and a polarising field bump, the net SYNC gain falls in the acceleration and deceleration phases (-0.45 and -0.25 nats/bin on a 0.47 baseline), because coupling gain depends on the operating point (cells pushed away from 50/50), not only on J. So a flat or falling gain change (G3a as written) is not a test of J constancy, and G3b (epoch-specific weights) did not detect a doubling of J restricted to the 10-bin acceleration phase (its epochs are +/-500 ms).
  2. *Change.* G3a is replaced as the primary test by **G3a'**: per condition, fit M1 to all kept reaches, simulate 5 datasets of the same size with the same fields and the same constant J (exact Gibbs sampling, 40 sweeps), run the same M0/SYNC CV on them, and take the phase-minus-baseline change of the raw SYNC gain as the "J constant at this operating point" reference. G3a' = observed change minus reference change, per session, Wilcoxon vs 0, bootstrap CI, in nats/bin; family = {G2, G3a' x3, G3b} (still 5 tests). The null-subtracted G3a is kept as a descriptive table. Limitation: the reference inherits M1's pseudo-likelihood J, which can be biased in strongly coupled sessions.
  3. *Disclosure.* A timing benchmark of `fit_glms.py` (2 real conditions, 1 shuffle, nothing inspected, output deleted) was run before the synthetic validation. G3b keeps its pre-registered form and is reported with its power limit stated.
