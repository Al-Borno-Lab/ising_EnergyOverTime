# 05_decision_tree_prediction/

## What this answers

Can a decision-tree (or ensemble) classifier predict, from session-level features alone, whether a given session/stimulus will show a significant J-coupling ("arbitration winner") event — i.e. is the arbitration outcome itself predictable, or does it look effectively stochastic given available features?

## Scripts

- `code/build_decision_tree_dataset.py` — Builds a per-session/per-stimulus CSV where each row is one (session, stim) pair, with target `is_j_peak` (whether a significant J-coupling event occurred near the kinematic window) plus model-quality, kinematic, and neural-statistic features. Run: `python build_decision_tree_dataset.py --data_folder <path> --output <csv_path>` (see `--window`, `--stim_min/--stim_max_exclusive`, `--sessions`, `--rep_start/--rep_end_exclusive`, `--peak_threshold`, `--smooth_sigma`, `--workers`, `--quiet` for tuning detection/scope).
- `code/run_decision_tree.py` — Trains a decision tree (or random forest via `--model`) on the built dataset to predict `is_j_peak`, then plots the tree/feature importances and reports accuracy with leave-one-session-out cross-validation. Run: `python run_decision_tree.py --csv ./notes/decision_dataset/decision_tree_dataset.csv` (see `--max_depth`, `--min_samples_leaf`, `--output_dir`, `--no_cv`, `--max_depth_plot`, `--exclude_cols`, `--n_estimators`, `--balance`, `--max_peak_ratio` for variants).
- `code/run_decision_tree_pipeline.sh` — Shell wrapper chaining the build and train steps into a single pipeline run.

## Key findings

No dedicated narrative write-up or analysis-note summary was found for this folder among the reviewed materials. `notes/decision_dataset/` contains multiple dataset-build and training run variants (CSVs, logs, tree/importance plots), but none were accompanied by a hand-written note summarizing conclusions. As a result, no grounded key findings can be reported here — this section should be filled in only after directly inspecting the run-specific tables/logs/plots in `notes/decision_dataset/`, rather than inferring outcomes from absent documentation.

## Caveats

- No narrative note exists to confirm whether the classifier's cross-validated accuracy indicates real predictive signal versus overfitting/chance performance given typical small session counts — this should be checked directly in the `notes/decision_dataset/` run outputs before drawing conclusions.
- Because dataset-build parameters (peak threshold, smoothing, stim/session ranges) vary across the multiple run variants in `notes/decision_dataset/`, results from different runs may not be directly comparable; confirm which parameter set underlies any figure before citing it.