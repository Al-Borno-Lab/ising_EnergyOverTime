# Code Audit Findings — Ising/Maximum-Entropy Cerebellar Analysis Pipeline

Audit method: every Python script in the reorganized project tree (29 files) was read —
either directly by hand (the core Ising-fitting/energy path, the P(K) collective-behavior
test, and peak-detection module) or via an LLM-assisted first pass targeting spin-encoding
mismatches, statistical-test misuse, CV leakage, off-by-one errors, circular reasoning, and
silently-swallowed exceptions. Every finding below marked **CONFIRMED** was independently
re-verified against the actual source and/or real data by direct inspection, grep, git
history, or an empirical test written for this audit — not accepted from the LLM pass alone.
Findings marked **NOT independently re-verified** are reported as-is from the LLM pass;
treat them as leads, not settled facts.

## Severity legend
- **HIGH / production impact** — affects results currently in use or planned for the paper.
- **HIGH / no production impact** — a real bug, but confined to dead/broken/never-run code.
- **MED / LOW** — real but narrower in scope, cosmetic, or a robustness concern rather than
  a correctness failure in current results.

---

## 1. CONFIRMED bugs with real production impact

### 1.1 `shared/src/arbitration.py` — `within_session_test` tests against the wrong kinematic reference
**Severity: HIGH.** The function's docstring and printed banners claim to answer "does energy
or firing rate have extrema closer to **max velocity**." The code extracts `max_vel_idx`
but never uses it — every distance (`energy_min_dist`, `firing_min_dist`, etc.) is computed
against `max_accel_idx` instead. Verified directly:
```
131:  energy_min_dist = abs(max_accel_idx - energy_min_idx)
146:  firing_min_dist = abs(max_accel_idx - firing_min_idx)
```
Worse, the companion plotting function draws its reference line at `x=max_accel_idx` but
labels it `f'Max Vel (idx={max_vel_idx})'` (lines 602-603, 638-639) — the line's position and
its printed index number both belong to acceleration, while the legend text and the number
displayed both say "velocity." **Every existing win/loss tally and figure from the
02_arbitration_energy_vs_firing pipeline is actually a test of closeness to peak
acceleration, not peak velocity, despite being documented, printed, and plotted as the
latter.** This needs to be relabeled/re-described (or the code fixed to genuinely use
`max_vel_idx`) before anything from this folder goes into the paper — the two kinematic
references are not interchangeable and any existing write-up phrased in terms of "velocity"
is currently mismatched to what was actually computed.

### 1.2 `arbitration_j_many.py` — `_xcorr_lag` sign convention is inverted relative to its own contract
**Severity: HIGH.** This is the function that answers the file's central question: "does J
lead or lag the kinematic event?" Its docstring states `lag > 0 → ts1 leads ts2`. I tested it
directly: with an impulse in `ts1` at index 3 and in `ts2` at index 5 (i.e., `ts1` genuinely
occurs first / leads), the function returns **-2**, not **+2**. Confirmed with the reverse
case too (returns **+2** when `ts1` actually lags). The sign is the exact opposite of the
documented and reported convention.

I checked git history: `_xcorr_lag` has had this exact formula since the function was first
introduced (commit `29c433a`, "Add J-peak arbitration analysis..."), and has never been
changed. **This means every `xcorr_lag_accel`/`xcorr_lag_vel` value in every
`results_j_kinematics.csv` / `session_summary_j_kinematics.csv` ever produced by this
pipeline has the opposite sign from what the reports/CSVs claim it means** — "J leads
acceleration" in the existing text reports is actually "J lags acceleration" and vice versa.
This is a one-line fix (`return (n - 1) - int(np.argmax(xcorr))`), but every existing
lead/lag conclusion drawn from this script needs to be re-read with the sign flipped, not
just fixed going forward.

### 1.3 `arbitration_j_many.py` — the currently committed script cannot execute at all
**Severity: HIGH bug, but affects future re-runs, not existing results (see below).**
`plt.rcParams.update(...)` is called at module level (line ~21) *before* `import
matplotlib.pyplot as plt` (line ~23) — confirmed by AST parse of the file, which shows the
`rcParams.update` statement executing at position 10, the `pyplot` import at position 11.
This raises `NameError: name 'plt' is not defined` on import; the script cannot run at all
in its current state. It also calls an undefined `_savefig` (defined only in three *other*
files — `visualization.py`, `peak_timing_analysis.py`, `pk_j_analysis.py` — none of which
this file imports from).

I checked whether this means the existing `results_j_kinematics.csv`/`session_summary_j_kinematics.csv`
files (found across multiple dated output folders, e.g. `arbitration_J/1/`,
`notes/J_analysis/J_analysis_arbitration_j_many/1/`) are corrupted by this: **no.** Git
history shows the `rcParams.update` block was added only in the most recent commit
(`6fe4fc7`, Aug 9 2026, "Tighten pk_j_analysis and model_quality_summary plot layouts") — a
styling change, apparently untested, that broke this file. All existing output files
predate that commit. **The existing J-lag results were generated by a working version of the
script; they are unaffected by this particular bug (though they ARE affected by 1.2 above).
But the script must be fixed before it can be re-run for anything — including re-running with
the sign fix from 1.2, or with the new figure-formatting convention you asked for.**

### 1.4 `05_decision_tree_prediction/run_decision_tree.py` — hyperparameter selection leaks into the reported CV score
**Severity: HIGH.** In `main()`, the `--balance` ratio sweep selects `best_ratio` by
maximizing leave-one-session-out (LOGO) balanced accuracy computed over *all* sessions. That
chosen ratio is then baked into the dataset and reused as `fold_ratio` in the "real" LOGO-CV
block later in the same function — the one whose accuracy is written into
`decision_tree_report.txt` and `cv_per_session.png`. This is hyperparameter selection and
performance evaluation performed on the same held-out folds with no outer/nested CV — a
textbook double-dipping pattern. The reported cross-validated accuracy is optimistically
biased by however much the ratio search improved fit; a nested-CV or held-out ratio-selection
scheme is needed before the reported accuracy can be trusted as an unbiased estimate.
(NOT independently re-verified beyond reading the code excerpt directly — the pattern is
mechanical and low-ambiguity, but I did not run the pipeline end-to-end to quantify the bias.)

### 1.5 `test_collective_j_timing.py` — selection bias in the primary (default) analysis path
**Severity: HIGH.** `build_dataset()`'s session-inclusion filter is:
```python
if not data["has_j_peak"] or data["j_peak_idx"] is None:
    continue
```
Inclusion is gated only on a significant J-coupling peak having been detected, while the
competing indicators (H, firing rate, energy) are independently gated/NaN'd elsewhere. This
guarantees `dist_j` is always present and derived from a peak the pipeline has already
certified "real," while competing indicators may be absent or noisier for the exact same
sessions — biasing the "closest indicator" tally and the pairwise Wilcoxon comparisons toward
J. A `build_dataset_raw_extrema` variant in the same file does NOT have this bias, but it is
not the function `main()` calls by default, and its results are not the ones reported first.
**Recommendation: make `build_dataset_raw_extrema` (or an equivalently unbiased
inclusion rule) the primary/headline comparison, and report the J-gated version only as a
secondary "conditional on J showing a peak" analysis.**

### 1.6 `shared/code/aggregate_transition_analysis.py` — p-hacked threshold selection
**Severity: HIGH.** `threshold_sweep()` computes pooled p-values for acceleration coupling
across 6 candidate thresholds and selects `best_thr = argmin(pooled_p)` — i.e., picks the
spike-defining threshold that minimizes the very p-value later reported as evidence of
significance. Separately, the CLI default (`threshold=2.0`) is documented in its own help
text as "the empirically best threshold for acceleration coupling" — meaning *ordinary,
non-sweep* runs already use a value apparently mined from the same dataset being tested. Any
p-value reported from this script (sweep or default) is not a valid significance test as
currently computed. This script also applies no multiple-comparisons correction across the
several tests per session × several sessions × several thresholds it runs (see full findings
file for detail), and pools sessions as independent when directory structure
(`<experiment>_rep1`, `_rep2`, ...) implies repeated measures from the same animal — both
compounding the p-hacking problem. **This script's outputs should not be used for any
significance claim in the paper without a substantial rewrite:** fix the threshold a priori
(or report the sweep transparently as exploratory), apply FDR/Bonferroni correction, and
either pool reps hierarchically or treat them as non-independent.

---

## 2. CONFIRMED — real bug, but NO impact on existing/production results

### 2.1 Dead-code spin-encoding bug (found earlier this session)
`calculate_energy_for_spike_data` and `analyze_neural_stimuli_wells` (in `analysis.py`) pass
un-converted {0,1} spins into `calc_e`, which expects {-1,+1}. Confirmed by grep: **zero
callers anywhere in the codebase.** The production path (`main.py` → `analyze_single_stimulus`
→ `calc_e_with_terms`) converts correctly.

### 2.2 `main-wells.py` / `main-wells-old.py` cannot run at all
`main-wells-old.py` imports `analyze_wells_data`, which no longer exists in `analysis.py`.
`main-wells.py` calls `load_wells_data(args.wells_file, args.mouse_id)` against a definition
requiring a third positional argument `layers` — `TypeError` on every invocation. Both are
legacy/abandoned entry points; neither could have produced any current result.

### 2.3 `visualization.py` — `k_corr`/model-quality encoding ambiguity does not trigger in practice
The LLM pass flagged that `plot_model_quality_summary`'s docstring claims to accept either
{0,1} or {-1,+1} input but only actually handles the {-1,+1} case correctly (a separate
helper, `_binary_spike_matrix`, explicitly normalizes for one sub-panel but not the main
correlation panels). I traced the only real caller: `main.py` always does
`bin_cat_p = 2 * preprocessingSpikes(...) - 1` (line 295) **before** passing `bin_cat_p` into
`plot_model_quality_summary` (line 132) — so in production this is always already {-1,+1}.
**No existing figure is corrupted by this.** It's still worth narrowing the docstring or
adding an explicit assertion, since the function silently mishandles {0,1} input if anyone
ever calls it that way.

### 2.4 `shared/src/processing.py` — fragile positional merge does not currently corrupt data
The LLM pass flagged `calculate_session_averages`'s velocity/acceleration merge (built via
`df.groupby(['reach_idx','stim'])`, then reassigned onto `df` by raw `.values` position) as a
row-misalignment risk if the input isn't already sorted to match groupby's iteration order.
**I tested this empirically** against a real `per_reach_state.csv` from the cluster
(`210421_rep1/full_reach/stim_0`, 14,000 rows, 20 reaches): `reach_idx` is already globally
monotonic non-decreasing and each per-stim CSV contains only one `stim` value, so
`groupby(['reach_idx','stim'])` reduces to grouping by `reach_idx` alone, in an order that
already matches the file's row order. Recomputing velocity_x via an explicit index-aligned
merge and comparing to the pipeline's actual positional-merge output gave **zero mismatches
across all 14,000 rows.** This is real production data, not a synthetic case, so I'm
confident the bug is currently dormant. It is still fragile-by-accident rather than
correct-by-design — recommend switching to an explicit `(reach_idx, stim)`-keyed merge as a
defensive fix, since nothing prevents a future upstream change (e.g. a differently-sorted
CSV writer) from silently triggering it.

---

## 3. Re-examined and DOWNGRADED from the initial LLM pass

### 3.1 `build_decision_tree_dataset.py` — J-window features vs. the peak label: weaker than "circular"
The LLM pass flagged `mean_j_in_window`/`std_j_in_window` as "near-deterministic functions of
the label-generating statistic," which would defeat the purpose of the decision tree. I read
`detect_j_peak`'s actual implementation (`_detect_peak_core`) line by line: the label's core
statistic (`level_z`) compares the window's peak value against `bg_mean`/`bg_std` computed
from **smoothed values outside the window** (`bg_mask[w_lo:w_hi] = False`) — a genuinely
different quantity from `mean_j_in_window`/`std_j_in_window`, which are the raw, unsmoothed
mean/std **inside** the window. These are not the same computation, so this is not literal
circularity. There is a real but softer correlation worth flagging: both quantities derive
from the same time window, so a decision tree that leans heavily on
`mean_j_in_window`/`std_j_in_window` for `is_j_peak` prediction is unsurprising and shouldn't
be over-interpreted as evidence the tree learned something about kinematics. **Recommendation:**
report feature importances both with and without these two J-window-summary columns to show
the tree isn't solely riding on them.

---

## 4. Other confirmed/plausible findings not yet independently re-verified

These come from the LLM-assisted pass and were not each individually re-checked by hand
(time-boxed against the scope of the full audit) — flagged here for follow-up, not yet
treated as settled:

- **`peak_near_max_velocity.py`**: default `min_prominence=0.0` means the peak-finder flags
  almost any local wiggle as a "detected" J peak, with no comparison to a null/surrogate
  distribution — this is exactly what the planned surrogate-null falsification experiment
  (next step) is designed to test properly.
- **Several bare `except Exception: ...`/`except: pass` blocks** silently convert real
  computation failures (malformed CSV, sklearn convergence failure, corrupt PDF chunk) into
  "missing data," indistinguishable from legitimately absent files, in:
  `utils.py::_load_hdf5_mat`, `aggregate_transition_analysis.py`'s logistic-regression block,
  `aggregate_model_quality_plots.py::load_csv`, `regen_model_quality_summary.py::main`,
  `generate_session_report.py`'s PDF-merge loop.
- **`collect_files.py`**: the filename-collision fallback retries exactly once (an `if`, not
  a loop); a second collision silently overwrites a previously-collected session's file with
  no warning.
- **`run_decision_tree.py`**: NaN imputation uses a single global column median computed
  before any train/test split, leaking held-out-session information into every LOGO-CV fold's
  imputed values.
- **`peak_timing_analysis.py`**: "closest feature" winner selection is biased toward whichever
  feature has the fewest missing values, and ties break deterministically toward
  earlier-listed features — both could distort the reported ranking of which signal is
  temporally closest to the acceleration peak.
- **`aggregate_model_quality_plots.py`**: P(K) padding assumes each session's `K` column
  starts at 0 and is contiguous; a session with a dropped/missing K-bin would be
  silently misaligned when averaged into the summary panel.

Full unabridged per-file LLM audit output is preserved in `llm_audit_raw.json` for reference.

---

## 5. Summary for the paper write-up

**Fix before writing anything from these pipelines:**
1. `arbitration.py` — decide whether the intended reference is acceleration or velocity, then
   fix the code/labels/docstring to match (§1.1).
2. `arbitration_j_many.py` — fix `_xcorr_lag`'s sign (§1.2) and the import-order/`_savefig`
   crash (§1.3) before re-running; re-interpret every existing lead/lag claim with the sign
   flipped in the meantime.
3. `run_decision_tree.py` — restructure the ratio sweep as nested/outer CV, or select the
   ratio on a held-out split (§1.4).
4. `test_collective_j_timing.py` — report the unbiased (`build_dataset_raw_extrema`) comparison
   as primary, not the J-gated one (§1.5).
5. `aggregate_transition_analysis.py` — needs a rewrite of its significance-testing logic
   before any p-value from it is usable (§1.6).

**No action needed** (confirmed dormant/dead): §2.1–2.4.

**Softened, not retracted:** §3.1 (decision-tree feature correlation, not circularity).
