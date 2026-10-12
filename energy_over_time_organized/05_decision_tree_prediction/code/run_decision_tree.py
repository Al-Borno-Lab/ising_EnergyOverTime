#!/usr/bin/env python
# coding: utf-8
"""
Train a decision tree to predict whether a J-peak occurs in a session/stim,
then visualise the full tree with rules, accuracy, and feature importances.

Usage
-----
    python run_decision_tree.py --csv ./notes/decision_dataset/decision_tree_dataset.csv

Optional flags
    --max_depth N         Cap tree depth (default: None = fully grown)
    --min_samples_leaf N  Minimum samples per leaf (default: 2)
    --output_dir PATH     Where to save output PNGs (default: same folder as CSV)
    --no_cv               Skip leave-one-session-out cross-validation
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)
from sklearn.model_selection import LeaveOneGroupOut


# ---------------------------------------------------------------------------
# Feature columns (everything except the target and IDs)
# ---------------------------------------------------------------------------

TARGET  = "is_j_peak"

# Columns that are identifiers or derived from the target — excluded
EXCLUDE = {
    "session_id", "stim_number",
    "is_j_peak",
    "idx_j_peak",     # only present when there IS a peak → data leak
    "j_peak_ratio",   # detection score used to produce the target → data leak
}


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    # Drop feature columns that are entirely NaN
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # Fill remaining NaNs with column median (simple imputation)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return df, X, y, feature_cols


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _feature_display_names(cols):
    """Shorten column names to fit inside tree nodes."""
    rename = {
        "n_neurons":           "n_neurons",
        "idx_velocity_max":    "idx_vel_max",
        "vel_peak_value":      "vel_peak",
        "idx_acceleration_max": "idx_accel_max",
        "accel_peak_value":    "accel_peak",
        "var_vel_after_max":   "var_vel_post",
        "var_accel_after_max": "var_accel_post",
        "r_ising":             "r_ising",
        "r_independent":       "r_indep",
        "r_ising_vs_indep":    "r_ising_vs_indep",
        "ising_indep_dist":    "ising_indep_dist",
        "baseline_fr":         "baseline_FR",
        "baseline_energy":     "baseline_E",
        "fr_delta":            "FR_delta",
        "energy_delta":        "E_delta",
        "mean_j_in_window":    "mean_J_win",
        "std_j_in_window":     "std_J_win",
        # J-matrix structure
        "mean_abs_j_coupling": "mean_|J|",
        "frac_pos_j":          "frac_pos_J",
        "j_coupling_std":      "std_J_matrix",
        "max_abs_j_coupling":  "max_|J|",
        # Criticality
        "critical_temperature": "T_c",
        "critical_energy":      "E_c",
        # External field
        "mean_abs_h":          "mean_|h|",
        "h_std":               "std_h",
    }
    return [rename.get(c, c) for c in cols]


def _build_clf(args):
    """Return a freshly constructed (unfitted) classifier matching --model."""
    if args.model == "forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features="sqrt",       # key RF trick — random feature subsets
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    return DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )


def plot_decision_tree(clf, feature_cols, output_dir, max_depth_plot=None):
    """Save a publication-quality decision tree figure."""
    display_names = _feature_display_names(feature_cols)

    depth      = clf.get_depth()
    n_leaves   = clf.get_n_leaves()
    # Scale figure width/height with tree size
    fig_w = max(20, min(60, n_leaves * 3.5))
    fig_h = max(8,  min(30, depth    * 3.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        clf,
        feature_names=display_names,
        class_names=["no peak", "peak"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        ax=ax,
        max_depth=max_depth_plot,
        fontsize=9,
    )
    ax.set_title(
        f"Decision tree — predicting J peak\n"
        f"depth={depth}, leaves={n_leaves}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "decision_tree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tree figure → {path}")


def plot_feature_importance(clf, feature_cols, output_dir):
    """Horizontal bar chart of feature importances.
    For Random Forest, error bars show std across individual trees."""
    display_names = _feature_display_names(feature_cols)
    importances   = clf.feature_importances_

    # RF exposes per-tree importances via estimators_
    if hasattr(clf, "estimators_"):
        tree_imps = np.array([t.feature_importances_ for t in clf.estimators_])
        std_imps  = tree_imps.std(axis=0)
    else:
        std_imps = None

    order  = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.45)))
    colors = ["#4C72B0" if importances[i] > 0 else "#CCCCCC" for i in order]
    xerr   = std_imps[order] if std_imps is not None else None
    ax.barh(
        [display_names[i] for i in order],
        importances[order],
        xerr=xerr,
        color=colors,
        edgecolor="white",
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#333333"},
    )
    label = "Mean Gini importance ± std (across trees)" if std_imps is not None \
            else "Gini importance"
    ax.set_xlabel(label, fontsize=11)
    ax.set_title("Feature importances", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importances.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance → {path}")


def plot_confusion(y_true, y_pred, title, filename, output_dir):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["no peak", "peak"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix → {path}")


def plot_cv_per_session(cv_rows, output_dir):
    """Bar chart of per-session CV accuracy."""
    sessions   = [r["session"] for r in cv_rows]
    accuracies = [r["accuracy"] for r in cv_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(sessions) * 0.6), 4))
    colors = ["#2ecc71" if a >= 0.7 else "#e74c3c" for a in accuracies]
    ax.bar(sessions, accuracies, color=colors, edgecolor="white")
    ax.axhline(np.mean(accuracies), color="navy", lw=1.5, ls="--",
               label=f"mean = {np.mean(accuracies):.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (held-out session)")
    ax.set_xlabel("Session (held-out)")
    ax.set_title("Leave-one-session-out cross-validation accuracy",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "cv_per_session.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CV per-session → {path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(clf, feature_cols, X, y,
                 train_acc, cv_rows, output_dir):
    display_names = _feature_display_names(feature_cols)
    name_map = dict(zip(feature_cols, display_names))

    is_forest = isinstance(clf, RandomForestClassifier)
    model_label = (f"Random Forest ({clf.n_estimators} trees)"
                   if is_forest else "Decision Tree")

    lines = [
        f"{model_label} Report — J peak prediction",
        "=" * 55,
        "",
        f"Samples        : {len(y)}",
        f"Features used  : {len(feature_cols)}",
    ]
    if not is_forest:
        lines += [
            f"Tree depth     : {clf.get_depth()}",
            f"Leaves         : {clf.get_n_leaves()}",
        ]
    else:
        lines += [
            f"n_estimators   : {clf.n_estimators}",
            f"max_depth      : {clf.max_depth}",
        ]
    lines += [
        "",
        f"Training accuracy (all data): {train_acc:.3f}",
    ]

    if cv_rows:
        cv_accs = [r["accuracy"] for r in cv_rows]
        lines += [
            "",
            "Leave-one-session-out CV",
            "-" * 30,
            f"  Mean accuracy : {np.mean(cv_accs):.3f}",
            f"  Std accuracy  : {np.std(cv_accs):.3f}",
            f"  Min accuracy  : {np.min(cv_accs):.3f}  (session {cv_rows[np.argmin(cv_accs)]['session']})",
            f"  Max accuracy  : {np.max(cv_accs):.3f}  (session {cv_rows[np.argmax(cv_accs)]['session']})",
            "",
            "Per-session accuracy:",
        ]
        for r in sorted(cv_rows, key=lambda x: x["session"]):
            lines.append(f"  {r['session']}  acc={r['accuracy']:.2f}  "
                         f"(n={r['n_test']}  peak_rate={r['peak_rate']:.2f})")

    lines += [
        "",
        "Feature importances (Gini)",
        "-" * 30,
    ]
    order = np.argsort(clf.feature_importances_)[::-1]
    for i in order:
        lines.append(f"  {display_names[i]:<22}  {clf.feature_importances_[i]:.4f}")

    if not is_forest:
        lines += [
            "",
            "Full tree rules",
            "-" * 30,
            export_text(clf, feature_names=list(name_map.values())),
        ]
    else:
        lines += [
            "",
            "Note: Random Forest has no single rule set.",
            "      See feature_importances.png for averaged feature contributions.",
        ]

    lines += [
        "",
        "Classification report (training set)",
        "-" * 30,
        classification_report(y, clf.predict(X),
                              target_names=["no peak", "peak"]),
    ]

    report = "\n".join(lines)
    path   = os.path.join(output_dir, "decision_tree_report.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Text report → {path}")
    print()
    print(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Decision tree for J-peak prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True,
                   help="Path to decision_tree_dataset.csv produced by "
                        "build_decision_tree_dataset.py.")
    p.add_argument("--max_depth", type=int, default=None,
                   help="Maximum tree depth (None = fully grown).")
    p.add_argument("--min_samples_leaf", type=int, default=2,
                   help="Minimum samples required at a leaf node.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for figures and report "
                        "(defaults to same folder as CSV).")
    p.add_argument("--no_cv", action="store_true",
                   help="Skip leave-one-session-out cross-validation.")
    p.add_argument("--max_depth_plot", type=int, default=None,
                   help="Limit tree visualisation to this depth "
                        "(useful for very deep trees).")
    p.add_argument("--exclude_cols", nargs="*", default=None,
                   metavar="COL",
                   help="Column names to exclude from the feature set. "
                        "E.g. --exclude_cols std_j_in_window mean_j_in_window")
    p.add_argument("--model", choices=["tree", "forest"], default="tree",
                   help="Model type: 'tree' (single decision tree, interpretable "
                        "rule set) or 'forest' (random forest — better "
                        "generalisation, reliable feature importances, no rule "
                        "set). (default: tree)")
    p.add_argument("--n_estimators", type=int, default=500,
                   help="Number of trees in the random forest (ignored for "
                        "--model tree). (default: 500)")
    p.add_argument("--balance", action="store_true",
                   help="Undersample the majority class so peaks and no-peaks "
                        "are equal in size before training. Helps when the "
                        "tree is over-predicting peaks due to class imbalance.")
    p.add_argument("--max_peak_ratio", type=float, default=None,
                   metavar="R",
                   help="Cap the peak:no-peak ratio at R (e.g. 1.5 allows at "
                        "most 1.5x more peaks than no-peaks). Ignored if "
                        "--balance is set. E.g. --max_peak_ratio 1.5")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path   = os.path.abspath(os.path.expanduser(args.csv))
    output_dir = args.output_dir or os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {csv_path} …")
    df, X, y, feature_cols = load_data(csv_path)

    # ── Apply user-requested column exclusions ─────────────────────────────
    if args.exclude_cols:
        unknown = [c for c in args.exclude_cols if c not in X.columns]
        if unknown:
            print(f"  [WARN] --exclude_cols: unknown column(s) ignored: {unknown}")
        drop = [c for c in args.exclude_cols if c in X.columns]
        if drop:
            print(f"  Excluding columns: {drop}")
            X = X.drop(columns=drop)
            feature_cols = [c for c in feature_cols if c not in drop]

    n_peaks    = int(y.sum())
    n_no_peaks = int((y == 0).sum())
    print(f"  {len(df)} rows, {len(feature_cols)} features  |  "
          f"peaks: {n_peaks}  no-peaks: {n_no_peaks}  "
          f"ratio: {n_peaks/max(n_no_peaks,1):.2f}:1")

    # ── Class balancing ────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    def _apply_ratio(X_, y_, ratio):
        """Undersample peaks so peak:no-peak <= ratio. Returns (X_, y_) copies."""
        n_no = int((y_ == 0).sum())
        max_pk = max(1, int(np.floor(ratio * n_no)))
        pk_idx = np.where(y_ == 1)[0]
        if len(pk_idx) <= max_pk:
            return X_, y_
        keep = rng.choice(pk_idx, size=max_pk, replace=False)
        sel  = np.sort(np.concatenate([keep, np.where(y_ == 0)[0]]))
        return X_.iloc[sel].reset_index(drop=True), y_.iloc[sel].reset_index(drop=True)

    chosen_ratio = None
    if args.balance and "session_id" in df.columns:
        # Auto-sweep: find the peak:no-peak ratio that maximises CV balanced
        # accuracy, then apply it to the full dataset before final training.
        natural_ratio = n_peaks / max(n_no_peaks, 1)
        max_ratio     = min(natural_ratio, 4.0)
        # Build candidate grid: 1.0 to max_ratio in 0.25 steps, always include
        # 1.0 (strict equality) and the natural ratio (no undersampling).
        sweep = sorted(set(
            [round(r, 2) for r in np.arange(1.0, max_ratio + 0.01, 0.25)]
            + [1.0, round(natural_ratio, 2)]
        ))

        print(f"  --balance: sweeping ratios {sweep} via leave-one-session-out CV …")
        groups   = df["session_id"].values
        logo     = LeaveOneGroupOut()
        best_bac, best_ratio = -1.0, 1.0

        for ratio in sweep:
            y_true_all, y_pred_all = [], []
            for tr_idx, te_idx in logo.split(X, y, groups):
                X_tr_r, y_tr_r = _apply_ratio(X.iloc[tr_idx], y.iloc[tr_idx], ratio)
                cv_clf = DecisionTreeClassifier(
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced",
                    random_state=42,
                )
                cv_clf.fit(X_tr_r, y_tr_r)
                y_pred = cv_clf.predict(X.iloc[te_idx])
                y_true_all.extend(y.iloc[te_idx].tolist())
                y_pred_all.extend(y_pred.tolist())
            bac = balanced_accuracy_score(y_true_all, y_pred_all)
            print(f"    ratio {ratio:.2f}  →  CV balanced-acc = {bac:.3f}")
            if bac > best_bac:
                best_bac, best_ratio = bac, ratio

        chosen_ratio = best_ratio
        print(f"  → Best ratio: {best_ratio:.2f}  (CV balanced-acc = {best_bac:.3f})")
        n_no  = int((y == 0).sum())
        max_pk = max(1, int(np.floor(best_ratio * n_no)))
        pk_idx = np.where(y == 1)[0]
        npk_idx = np.where(y == 0)[0]
        if len(pk_idx) > max_pk:
            keep = rng.choice(pk_idx, size=max_pk, replace=False)
            sel  = np.sort(np.concatenate([keep, npk_idx]))
        else:
            sel  = np.arange(len(y))
        X  = X.iloc[sel].reset_index(drop=True)
        y  = y.iloc[sel].reset_index(drop=True)
        df = df.iloc[sel].reset_index(drop=True)
        print(f"  Final dataset: {int(y.sum())} peaks / "
              f"{int((y==0).sum())} no-peaks ({len(y)} total)")

    elif args.balance:
        # No session_id available — fall back to hard 1:1
        X, y = _apply_ratio(X, y, 1.0)
        chosen_ratio = 1.0
        print(f"  --balance (no sessions): downsampled to {int(y.sum())} peaks / "
              f"{int((y==0).sum())} no-peaks ({len(y)} total)")

    elif args.max_peak_ratio is not None:
        chosen_ratio = args.max_peak_ratio
        X_new, y_new = _apply_ratio(X, y, args.max_peak_ratio)
        if len(y_new) < len(y):
            # rebuild df to match the reduced index
            n_no = int((y == 0).sum())
            max_pk = max(1, int(np.floor(args.max_peak_ratio * n_no)))
            pk_idx  = np.where(y == 1)[0]
            npk_idx = np.where(y == 0)[0]
            keep = rng.choice(pk_idx, size=min(max_pk, len(pk_idx)), replace=False)
            sel  = np.sort(np.concatenate([keep, npk_idx]))
            X, y = X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)
            df   = df.iloc[sel].reset_index(drop=True)
            print(f"  --max_peak_ratio {args.max_peak_ratio}: kept {int(y.sum())} peaks / "
                  f"{int((y==0).sum())} no-peaks ({len(y)} total)")
    else:
        chosen_ratio = None

    # ── Train on full (possibly balanced) dataset ──────────────────────────
    print(f"  Model: {args.model}"
          + (f"  (n_estimators={args.n_estimators})" if args.model == "forest" else ""))
    clf = _build_clf(args)
    clf.fit(X, y)
    train_acc = accuracy_score(y, clf.predict(X))
    print(f"  Training accuracy: {train_acc:.3f}")

    # ── Leave-one-session-out CV ───────────────────────────────────────────
    cv_rows: list[dict] = []
    if not args.no_cv and "session_id" in df.columns:
        groups = df["session_id"].values
        logo   = LeaveOneGroupOut()
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # Apply same balancing inside each CV fold (training split only).
            # chosen_ratio is set by --balance auto-sweep; args.max_peak_ratio
            # is the manual override; either way we use _apply_ratio.
            fold_ratio = chosen_ratio if chosen_ratio is not None else args.max_peak_ratio
            if fold_ratio is not None:
                X_tr, y_tr = _apply_ratio(
                    X_tr.reset_index(drop=True),
                    y_tr.reset_index(drop=True),
                    fold_ratio,
                )

            cv_clf = _build_clf(args)
            cv_clf.fit(X_tr, y_tr)
            y_pred = cv_clf.predict(X_te)

            session_name = str(groups[test_idx[0]])
            acc = accuracy_score(y_te, y_pred)
            cv_rows.append({
                "session":    session_name,
                "accuracy":   acc,
                "n_test":     len(y_te),
                "peak_rate":  float(y_te.mean()),
            })
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(y_pred.tolist())

        print(f"  CV mean accuracy: {np.mean([r['accuracy'] for r in cv_rows]):.3f}")
        plot_confusion(all_y_true, all_y_pred,
                       "Confusion matrix (leave-one-session-out CV)",
                       "confusion_matrix_cv.png", output_dir)
        plot_cv_per_session(cv_rows, output_dir)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    if args.model == "tree":
        plot_decision_tree(clf, feature_cols, output_dir,
                           max_depth_plot=args.max_depth_plot)
    else:
        print("  Tree figure skipped (random forest — no single rule set)")
    plot_feature_importance(clf, feature_cols, output_dir)
    plot_confusion(y, clf.predict(X),
                   "Confusion matrix (training set)",
                   "confusion_matrix_train.png", output_dir)

    # ── Text report ────────────────────────────────────────────────────────
    write_report(clf, feature_cols, X, y, train_acc, cv_rows, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
