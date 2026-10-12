# Decision-Tree Stage: Verification Findings

Scope: `build_decision_tree_dataset.py` and `run_decision_tree.py`, which train a
decision tree / random forest to predict `is_j_peak` (whether a coupling-strength
peak occurred in a session/stim's analysis window) from kinematic, model-quality,
and J-coupling-matrix features. This step re-examines a leakage concern flagged in
the prior code audit and independently re-runs the pipeline on real data.

## 1. Feature-leakage concern: downgraded, not dismissed

The target `is_j_peak` and two candidate features (`mean_j_in_window`,
`std_j_in_window`) are all computed from the exact same array slice `j_ts[lo:hi]`
in `build_decision_tree_dataset.py`. Reading `detect_j_peak`'s implementation in
`shared/src/peak_detection.py` confirms the detector does **not** use that
window's own mean/std as its criterion — it compares the window's smoothed local
maximum against a **background computed outside the window** (a prominence ratio
against background peaks, or a level-shift z-score against the pre-window
baseline). So `mean_j_in_window`/`std_j_in_window` are not a deterministic
restatement of the target — but they are still likely correlated with it (a
window containing a genuine peak will often also have an elevated mean/std),
so a tree that leans on them is not demonstrating real explanatory power from
independent evidence.

The pipeline's own author had already recognized the sharper form of this risk:
`run_decision_tree.py` hard-excludes `idx_j_peak` and `j_peak_ratio` (the
detector's own index/score) as flagged direct leaks, and separately exposes an
`--exclude_cols` flag specifically for dropping `mean_j_in_window`/
`std_j_in_window` if desired. Both a `no_j_stats` and a `yes_j_stats` variant
had already been run and saved under `notes/decision_dataset/`.

## 2. The bigger issue: none of the ~20 saved variants generalize across sessions

Rather than trust either saved run's numbers, all pre-computed report files were
downloaded from the cluster and the pipeline was independently re-run locally on
the current 42-row dataset (21 sessions × 2 stims with complete features; 28
peaks / 14 no-peaks). **Every configuration — pulled from the researcher's own
20+ historical runs and independently reproduced from scratch — shows the same
pattern: high training accuracy paired with leave-one-session-out (LOSO)
cross-validated accuracy that sits at or below the trivial "always predict the
majority class" baseline.**

| Source | Variant | n | Train acc. | LOSO-CV mean ± std | Majority baseline | CV below baseline? |
|---|---|---|---|---|---|---|
| researcher (n=63) | No J-window stats | 63 | 0.873 | 0.444 ± 0.376 | 0.730 | **yes** |
| researcher (n=63) | With J-window stats | 63 | 0.921 | 0.540 ± 0.333 | 0.730 | **yes** |
| researcher (n=42) | Leak-excluded, depth 3 | 42 | 0.881 | 0.500 ± 0.309 | 0.667 | **yes** |
| researcher (n=35, balanced) | Leak-excl., balanced 1:1 | 35 | 0.914 | 0.575 ± 0.396 | 0.500 | no (barely) |
| researcher (n=35, balanced) | Random forest, balanced | 35 | 1.000 | 0.575 ± 0.455 | 0.500 | no (barely) |
| independent rerun (n=42) | Tree w/ J-window stats | 42 | 1.000 | 0.595 ± 0.426 | 0.667 | **yes** |
| independent rerun (n=42) | Tree w/o J-window stats | 42 | 0.929 | 0.524 ± 0.422 | 0.667 | **yes** |
| independent rerun (n=42) | Random forest, all feats | 42 | 1.000 | 0.595 ± 0.426 | 0.667 | **yes** |

Six of eight variants have CV accuracy strictly below the majority-class
baseline; the two exceptions (balanced 1:1 subsets) beat a 50% coin flip by only
7–8 points, well within one CV standard deviation. Excluding the
`mean_j_in_window`/`std_j_in_window` features (addressing concern #1 above)
makes essentially no difference to this picture — CV accuracy drops slightly
(0.595 → 0.524 in the independent rerun), confirming those features were never
the reason the tree looked good; the reason was overfitting to only 21 sessions
× 2–3 samples each.

Per-session breakdown (independent rerun, tree with J-window stats, representative
of the pattern across all variants): of 21 held-out sessions, 5 score exactly 0.0,
6 score 0.5, and 10 score 1.0 — a bimodal, unstable pattern consistent with a tree
memorizing session-specific idiosyncrasies rather than a signal that transfers.

## 3. Interpretation

The decision-tree/random-forest stage, as currently built, does **not**
demonstrate that kinematic or Ising-model features predict J-coupling peaks in a
way that generalizes to a new session. Training accuracy (0.87–1.00, reported
in every saved run) is not informative here — with 11–23 features fit to
35–63 samples grouped into only 15–21 independent sessions, near-perfect
training fit is expected regardless of any true signal. The leave-one-session-out
CV numbers the pipeline itself already computes and saves (in every run
inspected) are the correct generalization estimate, and they show no reliable
performance above chance. This is not a bug in the code — `run_decision_tree.py`
is well-built (correct grouped CV, explicit leak exclusions, documented
override flags, class-balancing sweep) — it is a **sample-size / signal
limitation**: with this few independent sessions, a tree-based model cannot be
shown to generalize, and the existing runs already say so once the CV column is
read rather than the training-accuracy column.

## 4. Recommendation

- Report LOSO-CV accuracy (already computed, already saved) as the headline
  number for this analysis, not training accuracy.
- Given CV accuracy near or below the majority-class baseline across every
  configuration tried, the decision-tree stage does not currently support a
  claim that these features predict J-peaks; it should be described as
  exploratory / negative, or dropped from paper-facing claims pending more
  sessions.
- If more sessions become available, re-run `run_decision_tree.py
  --exclude_cols mean_j_in_window std_j_in_window` (removing the softer
  leakage risk) and re-check whether LOSO-CV clears the majority baseline with
  a real margin (e.g., a 1 CV-std margin) before treating the result as
  positive evidence.

## Artifacts
- `decision_tree_generalization.png` — training vs. LOSO-CV accuracy across all
  8 configurations (top), per-session held-out accuracy for a representative
  model (bottom).
- `decision_tree_cv_summary.csv` — the summary table above.
