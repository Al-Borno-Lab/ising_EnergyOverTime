# Pre-registration: operating-point hypothesis (written 2026-10-07, before any real-data computation of H1–H5)

**Hypothesis.** Purkinje-cell (PC) couplings J are fixed over the reach. The reach-related change in firing moves neurons relative to p = 0.5. Fixed couplings have the most influence on joint population patterns when units are near p = 0.5, so coupling influence should rise when the population moves toward 0.5 and fall when it moves away.

Nothing below may be changed after real-data results are seen, except where a change is logged in the amendment section at the end, with its date and reason.

## 1. Data and inclusion
- 21 sessions / 63 conditions (session × block), spikes in 10-ms bins, binarized (count > 0), window rows 100:800 (T = 700 bins).
- **Primary:** reaches with `ReachS.exclude = 1` dropped (1092 reaches). **Sensitivity:** all 1249 reaches.
- **Alignment:** bin 400 = the segment centre = the 1-cm outward crossing / stimulation trigger. Verified on stimulated reaches: laser onset median bin 400.1, IQR 397.8–400.9. Per-reach peak velocity is not used as an anchor: in 28% of reaches the window maximum lies more than 300 ms from the crossing.
- All conditions are included. A sensitivity analysis restricted to N ≥ 5 is also reported.

## 2. Model
- Time-varying-field pairwise model per condition, with J fixed in time: P_t(s) ∝ exp(Σ_i h_i(t) s_i + ½ Σ_ij J_ij s_i s_j), s ∈ {−1, +1}. h_i(t) is expanded in a cubic B-spline basis (`tvfield_model.bspline_basis(400)`; 50-ms knots from −600 to +1600 ms around bin 400, 250 ms outside).
- **Fit:** pseudo-likelihood initialization, then **exact maximum likelihood** (moment matching: the model's time-resolved means and its time-pooled pairwise correlations equal the data's). Exact enumeration is used for N ≤ 20 (39 conditions, 13 sessions). For N > 20, Monte-Carlo maximum likelihood is used, validated in H0 before use. Plain pseudo-likelihood parameters are not used for any thermodynamic quantity: they fail moment matching in the 2022 sessions.

## 3. Quantities, computed at every bin t from the fitted (h(t), J)
1. **Multi-information per neuron (primary):** I(t)/N = [Σ_i H_b(m_i(t)) − S(t)] / N, where S(t) is the model entropy and m_i(t) its marginals (bits).
2. **Coupling covariance per neuron (secondary):** χ_J(t)/N = Σ_{i≠j} Cov_t(s_i, s_j) / N, i.e. the off-diagonal part of the susceptibility to a uniform field.
3. **Heat capacity per neuron (secondary):** Var_t(E)/N at T = 1.
4. **Operating point:** v(t) = mean_i p_i(t)(1 − p_i(t)), with p_i(t) the model marginal, plus each neuron's distance |p_i(t) − 0.5|.

## 4. Windows (relative to bin 400; 1 bin = 10 ms)
- Baseline: bins −100 to −40 (−1.0 to −0.4 s).
- **Reach: bins −20 to +20 (−200 to +200 ms). This is the primary window.**
- Late: bins +50 to +150 (exploratory only; it is cohort-confounded, see the results log).
- Δ = mean over the window − mean over baseline.

## 5. Tests (unit = session; condition values averaged within a session; n = 21)
- **H2, influence rises during the reach.** ΔI/N in the reach window > 0: two-sided Wilcoxon signed-rank test on session means. Also a mixed model ΔI ~ 1 + (1 | mouse) over the 6 mice.
- **Null.** Trial-shuffle (each neuron's reach order permuted independently, 20 shuffles), refit, ΔI_null. The real session ΔI must exceed the 95th percentile of its own null.
- **H3, model-free confirmation and fixed-J adequacy.** A frozen-field model with fixed J is stationary, so its ΔI is 0 by construction; comparing against it would be circular. Within the fixed-J model, ΔI can only come from h(t). The empirical questions are therefore whether the data change the way that model predicts, and whether J really is fixed.
  - (a) Empirical time-locked covariance: c_ij(t) = Cov across reaches of (s_i(t), s_j(t)), with the time-locked mean removed so rate modulation is excluded. It is summarized as Σ_{i≠j} c_ij(t)/N, then Δ (reach − baseline) is taken, after debiasing for the number of reaches. This is compared with the fixed-J prediction Δχ_J/N from the full-data fit. The covariance inside the reach window is not a fitted moment: the fit matches only time-pooled correlations, and the window is 6% of bins. Sensitivity: the prediction averaged over the 5 fold fits, each trained without that fold's reaches. Test: per-session agreement in sign, and the Spearman correlation across conditions between the empirical and predicted Δ. The frozen-field prediction (Δ = 0) serves as the reference model.
  - (b) Epoch-specific J in the reach window (exact maximum likelihood, the same folds) must not improve held-out likelihood (Wilcoxon, p > 0.05, with its effect size reported). If it does improve, the change in influence is at least partly a change in J, and the "fixed J" part of the claim fails.
- **H4, mechanism.**
  - (a) Across conditions: Spearman ρ between ΔI/N and Δv (the operating-point change), plus a mixed model ΔI ~ Δv + (1 | session).
  - (b) Within conditions: neuron-level coupling influence (each neuron's conditional mutual information with the rest of the population, MI_i(t)) against Δ|p_i − 0.5|. The prediction is that neurons moving toward 0.5 gain influence and neurons moving away lose it. About 30% of neurons have baseline p < 0.5, which provides the internal contrast.
- **H5, nuclei-relevant output.** The coupling-driven excess of synchronous pauses: P(at least k of N silent | J) − P(at least k silent | J = 0, same marginals), with k = ⌈N/2⌉. Δ in the reach window, Wilcoxon test.
- Multiple comparisons: BH-FDR across the primary family {H2, H3, H4a, H5}.

## 6. Go / no-go
- **Supported:** H0 passes, AND H2 shows ΔI > 0 (p_FDR < 0.05, above the null), AND H3 holds, AND H4a has ρ > 0 (p_FDR < 0.05).
- **Refuted:**
  - ΔI ≤ 0;
  - or ΔI exists but is not related to Δv (H4a null);
  - or the empirical time-locked covariance does not change as predicted (H3a), or epoch-specific J improves prediction (H3b);
  - or the effect is present only in the 2022 cohort, i.e. present in 2022 and absent in 2021 when tested separately.
- **Fallbacks if refuted:** H5 alone (coupling-driven synchronous pauses), or stimulation invariance (step 10).

## 7. H0, synthetic validation (must pass before H1–H5 run on real data)
- Synthetic populations: N = 11 and 15 (exact; real fitted J matrices), N = 24 (Monte Carlo), R = 20 and 60 reaches, T = 700, 5 replicates.
- **Ground truth:** J taken from a real fitted condition (rescaled where stated). Fields are designed so that the marginals follow a reach-locked dip; samples are drawn exactly in each bin.
- **Scenarios:**
  - (A) fixed J, marginals dip from about 0.62 toward 0.45;
  - (B) fixed J, marginals rise from about 0.6 to 0.8 (away from 0.5);
  - (C) J × 1.6 inside the reach window, fields constant;
  - (D) J = 0, same dip as A;
  - (E) mixed population, with 30% of neurons starting below 0.5.
- **Pass criteria:**
  - (i) fitted J recovers the truth (r ≥ 0.9 at R = 60);
  - (ii) estimated ΔI has the correct sign in A (+), B (−) and D (≈0), and lies within 50% of the true ΔI;
  - (iii) in A, the fixed-J predicted Δχ_J matches the empirical time-locked covariance change (same sign; within the 95% CI over replicates); in C, the fixed-J model underpredicts it and the epoch-specific-J test detects the change;
  - (iv) in E, the neuron-level relation in H4b has the predicted sign;
  - (v) the shuffle null gives ΔI ≈ 0 in all scenarios.

## Amendments
1. **2026-10-07, before any real-data computation: populations with N > 20.** Monte-Carlo maximum likelihood is replaced by exact analysis of random 16-neuron subsets (10 subsets per condition, the same subsets for real data and the shuffle nulls). Quantities are reported per neuron and averaged over subsets. *Reason:* this keeps every quantity exact and uses the code already validated in H0, rather than adding a second, separately validated estimator. *Check:* in the four sessions with N = 17–19, the subset estimate is compared with the full-population estimate (Spearman ρ of ΔI/N across conditions, and sign agreement). Multi-information grows with N, so subset and full values are compared only within the same N. The session-level test uses Δ within each condition, which is what makes the different N comparable.
2. **2026-10-07, after H0 (synthetic only), before any real-data computation: H3b test.** Epoch-specific J (one free change per pair) did not detect a true 1.6× coupling increase at N = 15 (held-out gain −0.064 with 20 reaches, −0.007 with 60), although it did at N = 11 with 60 reaches (+0.041). It is replaced by a single coupling-scale parameter κ in the reach window, J_eff = (1 + κ)J, with the per-pair version kept as a secondary analysis. Test: κ̂ ≠ 0 across sessions (Wilcoxon), plus the held-out gain of the κ model. Requires a short synthetic check before use (scenarios A and C).
3. **Same date, same status: H4b neuron-level test.** In H0 scenario D (no coupling at all), the estimated neuron-level ρ is still −0.48 to −0.65. Noise in the fitted J produces spurious influence that grows as p approaches 0.5. H4b is therefore redefined as ρ_real − ρ_null, using the same trial-shuffle refits as the H2 null. The prediction is ρ_real − ρ_null < 0. This also requires a synthetic check (scenario D must give about 0, scenario E < 0).
4. **Same date, same status: H5 metric.** P(≥ ⌈N/2⌉ silent | J) − P(same | J = 0) falls in scenario A, where the hypothesis is true by construction (true Δ −0.017 at N = 15 with 60 reaches): the threshold sits where the dip moves the mean silent count. H5 is redefined as the coupling-driven excess **variance** of the silent count, Var_J(K) − Var_ind(K) = χ_J / 4. That makes H5 a restatement of the χ_J part of H2, so it is reported as a readout of H2 and dropped from the primary family. The primary family becomes {H2, H3, H4a}.

**H0 outcome (synthetic; see h0_summary_by_condition.csv and h0_synthetic_validation.png).**
- (i) J recovery: r = 0.96–0.99. Pass.
- (ii) ΔI correct in sign (40/40) and within 50% of the truth (40/40) in scenarios A and B; |ΔI| ≤ 6e-5 in D. Pass.
- (iii) The model-free covariance check separates C from A (pass). The per-pair epoch-J test is underpowered at N = 15, which leads to amendment 2.
- (iv) The neuron-level sign is recovered in E but biased in D, which leads to amendment 3.
- (v) Shuffle null |ΔI| ≤ 2e-4 everywhere. Pass.
5. **2026-10-12, before running the synthetic check of amendments 2-3 (criteria fixed here; outcome appended below after the run).** E10 step 1, `10_operating_point_h2_h4/code/step1_synthetic_amendments.py`. Synthetic only; J from `h0_truth_J.npz`; scenarios A, C, D, E; N in {11, 15}; R in {20, 60}; 5 replicates each; trial-shuffle null with 10 shuffles for the neuron-level test. κ is fitted by exact ML (W, J, κ jointly; J_eff = (1+κ)J inside bins 380-420) on all reaches; held-out gain from 5-fold CV.
   - **κ test (amendment 2).** Evaluated at R = 60 (R = 20 reported; real conditions have about 17 reaches, so R = 20 power is reported for interpretation, not as the gate). *Pass:* in A, the 95% t-interval of κ̂ over the 5 replicates contains 0 and |mean κ̂| < 0.15, for both N; in C (true κ = 0.6), mean κ̂ lies in [0.3, 0.9], the interval excludes 0, and held-out κ-model gain in the reach window is > 0 in at least 4 of 5 replicates, for both N.
   - **Neuron-level test (amendment 3).** ρ_real − ρ_null per replicate, ρ = Spearman(Δ|p−0.5|, ΔMI_i) over neurons, reach vs baseline. *Pass (R = 60, both N):* in D, |mean (ρ_real − ρ_null)| < 0.15 and the 95% t-interval contains 0; in E, mean < 0 and the interval excludes 0.
   - **Failure rule.** If any pass criterion fails, step 2 (real data) does not run; the failure is reported to the orchestrator and the affected test is dropped from the primary family or redefined by a further dated amendment before any real-data computation.

**Amendment 5 outcome (2026-10-12, synthetic; E10 `assets/tables/step1_summary.csv`, `assets/figures/step1_synthetic_amendments.png`, run in `10_operating_point_h2_h4/runs/RUN_LOG.md`).**
- **κ test (amendment 2): PASS.** A (truth 0), R = 60: κ̂ = −0.000 (95% interval −0.054 to 0.054) at N = 11 and −0.004 (−0.053 to 0.046) at N = 15. C (truth 0.6), R = 60: κ̂ = 0.602 (0.551 to 0.653) and 0.563 (0.535 to 0.592); held-out gain > 0 in 5/5 replicates at both N. At R = 20 (real-data-sized) C is also recovered (0.603, 0.653) and A stays near 0 (0.018, −0.078).
- **Neuron-level test (amendment 3): FAIL in scenario E.** ρ_real − ρ_null: D passes (0.050 at N = 11, −0.037 at N = 15; intervals contain 0), but E should be < 0 and is +0.087 (−0.188 to 0.362) at N = 11 and +0.073 (−0.061 to 0.207) at N = 15. Both ρ_real and ρ_null are strongly negative (about −0.6 to −0.8) in E, so the null reproduces the entire effect: the neuron-level relation is dominated by the estimation artefact, not by coupling.
- **Consequence under the failure rule:** step 2 (real data) is held. H4b (ρ_real − ρ_null) cannot be used as defined. Whether to drop it from the analysis or redefine it is decided by a further dated amendment before any real-data computation.
