# Session-Level Diagnostics and Outlier Review

Scope: all 21 sessions x 3 stim conditions = 63 processed conditions in
`outputs/energy_decomp_Jul_10_svg/` (9 sessions with raw `.mat` files available for independent
verification — see `raw_data_spotcheck_findings.md` — plus 12 sessions with derived outputs only).
For each condition: n_reaches, n_neurons, per-neuron firing rates, fitted h/J parameter distributions,
and per-timebin energy were pulled from `preprocessing_info.csv`, `h_parameters.csv`,
`J_parameters.csv`, and `per_reach_state.csv`. Full table: `session_diagnostics.csv`.

## Headline finding: a duplicate/near-duplicate neuron in session 210620

Flagging outliers by z-score across sessions on max|J|, mean J, energy-per-neuron, and mean firing
rate turned up **one dominant, unambiguous anomaly**: session **210620** has a fitted pairwise
coupling `J ≈ 1.96-1.98` between neurons 1 and 8 — roughly **5-10x larger** than the largest coupling
in any other session (typical max|J| across the other 20 sessions is 0.05-0.6), and this exact same
pair, at nearly the exact same magnitude, recurs identically across **all three stim conditions**
(1.957, 1.972, 1.980).

Investigating why: the raw spike trains for neurons 1 and 8 in this session are **99.2% identical
bin-for-bin**, with a Pearson correlation of **0.985** — versus a baseline of ~0.01 (max 0.27) for
20 randomly sampled neuron pairs in the same session, and versus a whole-dataset maximum of 0.65
for every *other* session's most-correlated pair. This is the signature of a **spike-sorting
duplicate**: the same physical unit almost certainly got split into two "neurons" during spike
sorting (or picked up redundantly on two channels), and the Ising fit is correctly detecting that
near-total redundancy — it manifests as an enormous pairwise coupling because two variables that are
almost always simultaneously on/off are indistinguishable from a maximum-entropy standpoint.

A systematic re-scan of the highest-correlated neuron pair in *every* session (`session_diagnostics.png`,
right panel) found a second, milder case of the same issue: session **220515** has a max
pairwise correlation of 0.89 (neurons 13/15) — also flagged independently as a J-magnitude outlier
(max|J|≈1.6). All other 19 sessions have max pairwise correlation ≤0.65, well below the range
consistent with duplicate units.

**Both flagged sessions (210620, 220515) fall in the group with no raw `.mat` file available**, so
this can't be cross-checked against the raw voltage traces from here — but the duplication is visible
directly in the already-processed `per_reach_state.csv` spike columns, independent of whether the
Ising-fitting code itself is correct.

## Why this matters

This is a data-curation issue upstream of the modeling code, not a bug in the Ising pipeline — but it
directly affects any conclusion that compares coupling strength (`J` magnitude) across sessions or
uses total/mean energy as a measure of collective synchrony, because these two sessions' J-magnitude
statistics are being inflated by one redundant unit rather than reflecting genuine population
structure. Recommended fix: check the spike-sorting output for sessions 210620 and 220515 for a
duplicated/split unit at the flagged neuron indices, merge or drop one copy, and refit before including
these sessions in any cross-session comparison of coupling magnitude.

## Other observations (not flagged as errors)

- Session 210425 (mean_J outlier, z>3) has only 5 neurons — small-N Ising fits are known to produce
  larger per-pair coupling estimates than large-N fits from the same underlying process; this is an
  expected statistical property of the fitting procedure at low N, not an anomaly, and 210425 has raw
  data and passed the bit-exact spot-check.
- Firing rates are broadly consistent across all 21 sessions (0.37-0.72 spikes/bin), with no session
  showing pathologically silent or saturated units.
- No session shows a NaN, negative-variance, or otherwise malformed h/J parameter file.

## Files

- `session_diagnostics.csv` — full 63-row diagnostic table (all sessions x stims).
- `session_diagnostics.png` — left: max|J| by session (raw-verified vs. derived-only, colored);
  right: highest pairwise spike-train correlation by session, with the duplicate-unit threshold marked.
