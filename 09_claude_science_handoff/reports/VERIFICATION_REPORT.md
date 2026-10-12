# VERIFICATION_REPORT.md
## Ising / Maximum-Entropy Cerebellar Pipeline — Verification & Paper-Prep Master Report

**Scope of this audit:** the `energy_over_time_script` project on `math-alderaan.ucdenver.pvt`,
covering the full pipeline from raw spike data to the paper's central claims: (1) a
line-by-line code audit of every script producing a reported number, (2) an independent
raw-data recomputation, (3) a new adversarial falsification experiment testing the
"collective behavior" premise itself, (4) a session-level diagnostic sweep, and (5) an
independent evaluation of the decision-tree stage. This document synthesizes all five,
lists what is safe to build on, what needs fixing first, and what remains open — for review
**before** any figure regeneration or manuscript drafting begins.

---

## 0. Framing: what this method actually is

The pipeline fits a pairwise Maximum-Entropy (Ising) model to binarized population spike
data and tracks the fitted energy/coupling terms alongside kinematics. Nothing in the
underlying method is specific to Purkinje cells or the cerebellum — it is a general
**collective-activity detector**: given any binned, binarized multi-neuron spike train, it
asks (a) whether population co-activity departs from what independently-firing neurons at
the same individual rates would produce (tested fresh in §3 below), and (b) whether a
pairwise-coupling model captures that departure, and how that captured structure evolves
relative to a behavioral event. The cerebellar motor-execution setting here is the
application, not a constraint on the method.

---

## 1. Code audit — what is safe to use, what must be fixed first

Full detail: `code_audit_findings.md`. 29 scripts read; every confirmed finding below was
independently re-verified against source, git history, or real data — not accepted from an
LLM-assisted first pass alone.

### Must fix before anything from these paths goes into the paper
1. **`arbitration.py` (`within_session_test`)** — every existing "closest to max velocity"
   result is actually computed against **max acceleration**; the plotting code even mislabels
   the acceleration-derived reference line as "Max Vel." Every win/loss tally and figure from
   `02_arbitration_energy_vs_firing` needs re-description (or the code fixed) before use.
2. **`arbitration_j_many.py` (`_xcorr_lag`)** — the lead/lag sign is inverted relative to its
   own documented contract, confirmed by a direct impulse test and present since the
   function's introduction. **Every existing "J leads/lags" conclusion has the opposite sign
   from what's reported** and must be re-read flipped, not just fixed going forward.
3. **`arbitration_j_many.py`** — the currently committed version cannot execute at all
   (rcParams set before the `matplotlib.pyplot` import, plus an undefined `_savefig` call).
   Existing outputs predate the breaking commit and are unaffected by this specific bug, but
   the script must be repaired before any re-run — including a re-run with the sign fix above.
4. **`run_decision_tree.py`** — the `--balance` ratio sweep selects its ratio by maximizing
   the same leave-one-session-out CV score later reported as the result (double-dipping, no
   outer/nested CV). Given §5 below already shows CV accuracy at or below chance regardless,
   this makes an already-negative result look less negative than it should.
5. **`test_collective_j_timing.py`** — the default analysis path only includes sessions with
   a detected J-coupling peak, while competing indicators (H, firing rate, energy) are
   independently gated elsewhere — biasing the "closest indicator" tally toward J. An unbiased
   alternative (`build_dataset_raw_extrema`) exists in the same file but is not the default;
   it should be the headline comparison.
6. **`aggregate_transition_analysis.py`** — its threshold sweep selects the spike-defining
   threshold by minimizing the very p-value later reported as significance (p-hacking by
   construction), with no multiple-comparisons correction and reps pooled as independent
   despite coming from the same animal. No significance claim from this script is currently
   usable without a rewrite.

### Confirmed but no production impact (dead/dormant code — no action needed)
- A spin-encoding bug in `analysis.py` has zero callers anywhere in the codebase.
- `main-wells.py` / `main-wells-old.py` cannot run at all (legacy, abandoned entry points).
- A model-quality-plot encoding ambiguity never triggers because the one real caller always
  pre-converts to {-1,+1}.
- A positional-merge fragility in `processing.py` was empirically tested against a real
  14,000-row `per_reach_state.csv` and produced zero mismatches — dormant today, worth
  hardening defensively.

### Downgraded, not retracted
- The decision-tree J-window-feature "circularity" concern is real but softer than first
  flagged — see §5, which supersedes this with a bigger, independently-confirmed finding.

### Leads not yet independently re-verified (see `code_audit_findings.md` §4 for the full list)
Includes a zero-prominence default in `peak_near_max_velocity.py` (addressed empirically by
the falsification experiment in §3), several silent `except: pass` blocks that convert real
failures into "missing data," a single-collision-retry bug in `collect_files.py`, CV-leaking
global-median NaN imputation in `run_decision_tree.py`, and a P(K) alignment assumption in
`aggregate_model_quality_plots.py`. None of these were exercised against real data in this
audit; treat as follow-up leads.

---

## 2. Raw-data spot-check — the foundational pipeline stage is verified correct

Full detail: `raw_data_spotcheck_findings.md`, `verification_spotcheck.csv`.

Every raw `.mat` session file that exists on the cluster (**9 of the 21 processed sessions**,
27 session×stim combinations) was independently re-derived from scratch — own MAT parsing,
own binarization, own spin encoding, own energy decomposition — without calling any audited
project code, and compared against the pipeline's saved outputs.

**Result: 27/27 exact matches** on the spike matrix (bit-exact), and on energy/J-term/h-term
(max abs. difference 1.07e-14, i.e. float64 rounding only). The most safety-critical stage of
the pipeline — raw spikes → spin encoding → energy decomposition, which feeds every
downstream figure and statistic — is fully verified correct against ground truth for every
session where ground truth is available.

**Open gap:** raw data could not be located on the cluster for **12 of the 21 sessions**
currently used in aggregate figures (210515, 210606, 210619, 210620, 210622, 210623, and the
six 2022 sessions 220515–220520) — only derived outputs exist for them today. This is not
evidence of a problem in those sessions, only that they are currently unauditable from this
cluster. **Recommend locating the original raw files before the paper reports results that
depend on them**, and re-running this same spot-check once found.

---

## 3. Surrogate-null falsification — the core "collective behavior" premise survives

Full detail: `falsification_findings.md`, `falsification_results.csv`, `falsification_test.png`.

This experiment assumed the opposite of the paper's premise (that observed neural co-activity
is fully explained by independently-firing neurons at their own individual rates) and tried to
reject that assumption on real data — the most literal adversarial test available, using
500 rate-preserving shuffles per condition as the null and a closed-form independent-neuron
`P(K)` as the point of comparison.

**Result: all 27/27 session×stim conditions (full raw-data coverage) reject the
independent-firing null decisively** — z-scores range from 4.9 to 116 (median 22), with every
condition's real discrepancy statistic exceeding all 500 of its own shuffles (empirical
p-value at the resolution floor, ≈0.002). Even the weakest-margin condition clears 4 standard
deviations.

**What this does and does not establish:** the population data used in this project shows
real, statistically decisive collective structure beyond independent firing — the basic
precondition for applying any maximum-entropy framework is met. It does **not** by itself
validate that the pairwise-only Ising model is the right or sufficient description of that
structure (versus needing higher-order terms, or fitting for the wrong reason) — that is a
separate, partially-addressed model-quality question. **Recommended follow-up** (not done
here, requires the coniii MCH fitter on the cluster): refit the Ising model on a subset of
these same surrogates and confirm fitted couplings collapse toward zero on surrogate data
while remaining large on real data.

---

## 4. Session-level diagnostics — one data-curation issue found, most sessions clean

Full detail: `session_diagnostics_findings.md`, `session_diagnostics.csv`, `session_diagnostics.png`.

Across all 21 sessions × 3 stims (63 conditions), z-scoring max|J|, mean J, energy-per-neuron,
and mean firing rate flagged **one dominant anomaly**: session **210620** has a pairwise
coupling J≈1.96–1.98 between two neurons — 5–10× the largest coupling anywhere else in the
dataset, recurring at nearly identical magnitude across all three of its stim conditions. The
raw spike trains for that pair are 99.2% bin-identical (r=0.985, versus ≤0.65 for every other
session's most-correlated pair) — the classic signature of a **spike-sorting duplicate**
(one physical unit split into two channels/units). A milder second case was found in session
**220515** (r=0.89, also a J-magnitude outlier). Both flagged sessions fall in the
raw-data-unavailable group (§2), so this cannot be cross-checked against original voltage
traces from here, but the duplication is directly visible in the already-processed spike
columns.

This is a data-curation issue upstream of the Ising-fitting code, not a modeling bug — but it
inflates J-magnitude statistics for these two sessions specifically and should be resolved
(merge/drop the duplicate unit, refit) **before either session is used in any cross-session
comparison of coupling magnitude.** One other flagged outlier (session 210425, low neuron
count N=5) is an expected statistical property of small-N Ising fits, not an anomaly, and
already passed the raw-data spot-check bit-exact. No session shows NaN/malformed parameters
or pathological firing rates.

---

## 5. Decision-tree stage — does not currently support a positive claim

Full detail: `decision_tree_findings.md`, `decision_tree_cv_summary.csv`, `decision_tree_generalization.png`.

The softer feature-correlation concern from §1 (downgraded item) was independently checked
against the peak-detector's actual implementation: it compares the window's peak against a
background computed **outside** the window, not the window's own mean/std, so it is not
literal circularity. Excluding the two suspect features barely changes the result, however,
because of a bigger, previously undocumented issue found by re-running the pipeline
(independently, on the current dataset) and by pulling ~20 of the researcher's own historical
run variants off the cluster:

**In 8 of 8 configurations checked (5 historical + 3 independently reproduced), training
accuracy is 0.87–1.00, but leave-one-session-out cross-validated accuracy is at or below the
trivial "always predict the majority class" baseline in 6 of 8, and only marginally above it
(within one CV standard deviation) in the remaining 2.** This holds regardless of whether the
suspect J-window features are included, and regardless of single-tree vs. random-forest.
With 11–23 features fit to only 15–21 independent sessions, near-perfect training fit is
expected and uninformative; the tree does not demonstrate that these features predict
J-coupling peaks in a way that generalizes to a new session.

**Recommendation:** report LOSO-CV accuracy, not training accuracy, as the headline number if
this analysis appears in the paper; as currently evidenced, describe it as exploratory/negative
rather than a positive finding, pending more independent sessions.

---

## 6. Figure-format spec (captured, not yet applied)

Full detail: `figure_format_spec.md`. The existing house convention under
`shared/results/comparisons/` — PNG+PDF+TXT file triplets, bold top-left `<row><col>` panel
labels with no in-plot legend (color-coded axis labels instead), a shared shaded-window +
dashed-event-time convention across time-series panels, and a fixed sidecar-`.txt` field
order/precision that becomes the caption — has been documented with concrete field-by-field
examples so that any figure regenerated after this audit is resolved can match it exactly.
**No figures have been regenerated as part of this audit**; the diagnostic figures produced
here (`falsification_test.png`, `session_diagnostics.png`, `decision_tree_generalization.png`)
are plain internal-review figures and intentionally do not follow the paper convention.

---

## 7. Numbered list of remaining holes and open questions

1. **Raw data missing for 12/21 sessions** (§2) — locate original files (backup, external
   drive, or original provider) before those sessions' results are reported as verified; they
   are not known to be wrong, only currently unauditable.
2. **Two sessions (210620, 220515) likely contain a duplicated spike-sorted unit** (§4) —
   needs resolution at the spike-sorting stage (which cannot be done from derived data alone
   for these two, since their raw files are among the missing 12) before either is used in a
   cross-session coupling-magnitude comparison.
3. **Every existing arbitration "velocity" claim is mislabeled acceleration, and every
   existing J-lag/lead sign is inverted** (§1, items 1–2) — these need correction/re-description
   before being cited in the paper; the underlying data is not necessarily wrong, but the
   existing text/labels describing it are.
4. **`arbitration_j_many.py` cannot currently run** (§1, item 3) — must be fixed before any
   re-run (e.g., after the sign fix, or under the new figure-format convention).
5. **Two significance-generating scripts need methodological rework before any p-value from
   them is citable**: `aggregate_transition_analysis.py` (p-hacked threshold selection, no
   multiple-comparisons correction, non-independent reps pooled as independent) and
   `run_decision_tree.py`'s balance-ratio sweep (hyperparameter selection leaking into the
   reported CV score) — though for the decision tree the CV score is already negative even
   with this optimistic bias (§5), so the qualitative conclusion is unlikely to flip, only
   become more negative.
6. **The default `test_collective_j_timing.py` comparison is selection-biased toward J** (§1,
   item 5); the unbiased `build_dataset_raw_extrema` variant in the same file should be
   promoted to the primary reported comparison.
7. **Decision-tree stage does not currently support a positive claim** (§5) — needs either more
   independent sessions or should be described as exploratory/negative in the paper.
8. **Generalizability beyond this cerebellar dataset is untested** — everything verified here
   (raw-data fidelity, the surrogate-null falsification, the diagnostic sweep) was run only on
   this project's Purkinje-cell recordings. The method itself is not cell-type- or
   region-specific (§0), but no non-cerebellar dataset was available in this audit to
   demonstrate that generality empirically; if the paper wants to claim general applicability,
   that requires a separate validation on at least one non-cerebellar collective-neural dataset.
9. **The falsification experiment (§3) validates that collective structure exists, not that
   the pairwise Ising model is the correct/sufficient description of it** — the recommended
   surrogate-refit follow-up (refitting J on surrogates and confirming couplings collapse
   toward zero) has not been run; it requires the coniii MCH fitter on the cluster and was out
   of scope for this audit's compute budget.
10. **Several leads from the audit were not independently re-verified against real data**
    (§1, "leads not yet independently re-verified") — silent exception handling that may mask
    real failures as missing data, a P(K) alignment assumption, and a CV-leaking NaN
    imputation scheme; worth follow-up but not yet confirmed as production-impacting.

---

## Summary table of deliverables

| Step | Document | Supporting files |
|---|---|---|
| Code audit | `code_audit_findings.md` | `llm_audit_raw.json` |
| Raw-data spot-check | `raw_data_spotcheck_findings.md` | `verification_spotcheck.csv` |
| Surrogate-null falsification | `falsification_findings.md` | `falsification_results.csv`, `falsification_test.png` |
| Session diagnostics | `session_diagnostics_findings.md` | `session_diagnostics.csv`, `session_diagnostics.png` |
| Decision-tree stage | `decision_tree_findings.md` | `decision_tree_cv_summary.csv`, `decision_tree_generalization.png` |
| Figure-format spec | `figure_format_spec.md` | — |
| **This report** | `VERIFICATION_REPORT.md` | — |

**Bottom line:** the foundational data pipeline (raw spikes → energy decomposition) is
verified bit-exact wherever raw data exists, and the core premise that this population shows
real collective structure beyond independent firing survives an adversarial falsification
test decisively. Against that, several specific downstream claims (arbitration
velocity/acceleration labeling, J-lag sign, two significance-generating scripts, the
decision-tree stage, and 12 sessions' raw-data provenance) need correction or further work
before they can support paper-facing claims as currently written. None of the findings above
undermine the central premise; they narrow which existing numbers and labels can be trusted
as-is versus need fixing or re-scoping first.
