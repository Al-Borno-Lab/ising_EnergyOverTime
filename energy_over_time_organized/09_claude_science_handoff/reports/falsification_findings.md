# Surrogate-Null Falsification Test: Is the "Collective Behavior" Real?

## Motivation

The project's central claim is that the cerebellar Purkinje population shows genuine **collective
(synchronous) structure** during motor execution — behavior a pairwise maximum-entropy (Ising) model
is built to capture. Before trusting any Ising-model output, the necessary precondition must hold:
population-level co-activity (`K` = number of neurons firing together in a time bin) must deviate from
what independent, uncorrelated neurons with the *same individual firing rates* would produce by chance.
If it doesn't, there is nothing "collective" to model, and the entire premise collapses.

This test assumes the opposite of the project's claim and tries to falsify that assumption: **assume
observed co-activity is explainable by independent firing, then see if the data can prove that assumption
wrong.**

## Method

For each of the 27 session x stim conditions (9 raw sessions x 3 stims, full raw-data coverage — same
data verified bit-exact in the raw-data spot-check):

1. Computed each neuron's empirical firing probability `p_i` from its own binarized spike train.
2. Computed the **independent-neuron null**, `P_indep(K)`, exactly via polynomial convolution of the
   per-neuron Bernoulli(`p_i`) distributions — the textbook closed-form prediction for K co-active
   neurons if there were truly zero interaction between them.
3. Computed the **observed** `P_data(K)` directly from the real binarized data.
4. Defined a discrepancy statistic `D = sum_K |P_data(K) - P_indep(K)|` (L1 distance).
5. Built an empirical **null distribution** for `D` by generating 500 surrogate datasets per condition:
   each neuron's spike train independently time-permuted (shuffled), which exactly preserves every
   neuron's own firing rate (so `P_indep` is unchanged) while completely destroying any true
   instantaneous cross-neuron correlation. Recomputed `D` for each shuffle.
6. Compared the real `D` against this null distribution: empirical p-value and z-score.

This directly tests whether the observed population synchrony could plausibly arise from independently
firing neurons with the same individual rates — the strongest and most literal null hypothesis available.

## Results

**All 27/27 session x stim conditions reject the independent-firing null outright.**

- z-scores range from **4.9 to 116** (median 22); every single condition's real discrepancy exceeds
  every one of its own 500 shuffles (empirical p-value at the resolution floor, ≈0.002, for all 27).
- The lowest-margin condition (210423, stim 1, z=4.9) is still >4 standard deviations beyond the null
  mean — nowhere close to marginal.

Full per-condition z-scores/p-values: `falsification_results.csv`. Figure: `falsification_test.png`
— left panel shows a representative session's `P(K)` visibly narrower/shifted relative to the
independence prediction; right panel shows every condition's z-score is far past the standard
rejection threshold (z=3, dashed line).

## Interpretation

**The premise survives the adversarial test.** The cerebellar Purkinje population data used in this
project shows real, statistically decisive collective co-activity structure that cannot be explained
by neurons firing independently at their own individual rates. This validates the basic motivation for
applying a maximum-entropy/Ising framework at all — there is genuine higher-order structure in this data
for such a model to characterize.

## Important scope limitation — this test does NOT validate the Ising model itself

This experiment establishes that **collective structure exists**; it does **not** establish that the
project's specific **pairwise-only Ising model** is an accurate or sufficient description of that
structure (versus, e.g., needing triplet/higher-order terms, or being right for the wrong reason). That
is a separate model-quality question the pipeline already partially addresses via its own
`P(K)`-comparison outputs (`model_quality_summary_P_K.csv` seen during the code audit, comparing the
*fitted* Ising `P_ising(K)` — not the naive independence prediction used here — against `P_data(K)`).

**Recommended follow-up** (not performed here due to compute cost — requires the coniii MCH fitting
routine on the cluster): refit the full pairwise Ising model on a subset of these same 500 surrogates
per session and confirm the fitted `J` couplings collapse toward zero on surrogate (independent-by-
construction) data while remaining large on real data. That would additionally validate that the
model-*fitting* procedure itself is not prone to hallucinating spurious couplings from uncorrelated
input — a distinct failure mode from the one ruled out here.
