#!/usr/bin/env python3
"""
peak_timing_analysis.py
-----------------------
Reads the CSV produced by pk_j_analysis.py (pk_j_summary.csv) and asks:

    Which kinematic / neural feature is most temporally proximal to the
    moment of peak acceleration?

Features compared:
    - FR peak        (fr_peak_idx)
    - FR trough      (fr_trough_idx)
    - J peak         (j_peak_idx)
    - H peak         (h_peak_idx)

For every session/stim row the lag to accel_peak_idx is computed as:

    lag_<feat> = <feat>_idx - accel_peak_idx

(positive = feature comes AFTER max acceleration)

Two sets of graphs are generated:
    1.  All sessions  (collective + non-collective)
    2.  Collective sessions only  (is_collective == True)

For each set:
    A) Bar chart of winner counts  — which feature had the smallest |lag|
    B) Violin / swarm plot of signed lag distributions per feature
    C) Grouped bar chart of mean |lag| ± SEM per feature

Usage
-----
    python peak_timing_analysis.py --csv ./notes/pk_analysis/pk_j_summary.csv \\
                                   --output_dir ./notes/peak_timing
    # restrict to stim 0 only:
    python peak_timing_analysis.py --csv ... --stim 0 --output_dir ...
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "axes.grid":        False,
    "font.size":        21,
    "axes.titlesize":   24,
    "axes.labelsize":   21,
    "xtick.labelsize":  19,
    "ytick.labelsize":  19,
    "legend.fontsize":  19,
    "figure.titlesize": 24,
})

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURES = [
    {"col": "fr_peak_idx",   "label": "FR peak",  "color": "limegreen"},
    {"col": "fr_trough_idx", "label": "FR trough", "color": "tomato"},
    {"col": "j_peak_idx",    "label": "J peak",    "color": "steelblue"},
    {"col": "h_peak_idx",    "label": "H peak",    "color": "firebrick"},
]


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _savefig(fig, path: str, **kwargs):
    fig.savefig(path, **kwargs)
    svg_path = os.path.splitext(path)[0] + ".svg"
    svg_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg_path, **svg_kwargs)


# ---------------------------------------------------------------------------
# Lag computation
# ---------------------------------------------------------------------------

def _lag_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row add columns:
        lag_<feat>      — signed lag (feat_idx - accel_peak_idx)
        abs_lag_<feat>  — |lag|
    Plus a 'winner' column with the feature label of the smallest |lag|.
    Returns only rows where at least one feature has a valid lag.
    """
    out = df.copy()
    lag_cols = []
    for feat in FEATURES:
        col = feat["col"]
        lag_col = f"lag_{col}"
        out[lag_col] = out[col].astype(float) - out["accel_peak_idx"].astype(float)
        out[f"abs_{lag_col}"] = out[lag_col].abs()
        lag_cols.append(f"abs_{lag_col}")

    # Drop rows where ALL abs lags are NaN
    out = out.dropna(subset=lag_cols, how="all").copy()
    out["winner"] = pd.array([None] * len(out), dtype=object)

    for idx in out.index:
        lags = {}
        for feat in FEATURES:
            v = out.at[idx, f"abs_lag_{feat['col']}"]
            if not (isinstance(v, float) and np.isnan(v)):
                lags[feat["label"]] = v
        if lags:
            winner = min(lags, key=lags.get)
            out.at[idx, "winner"] = winner

    return out


def _winner_counts(lag_df: pd.DataFrame) -> dict[str, int]:
    counts = {f["label"]: 0 for f in FEATURES}
    for label in lag_df["winner"].dropna():
        if label in counts:
            counts[label] += 1
    return counts


# ---------------------------------------------------------------------------
# Plot A — winner bar chart
# ---------------------------------------------------------------------------

def plot_closest_counts(lag_all: pd.DataFrame, lag_coll: pd.DataFrame,
                        output_dir: str):
    counts_all  = _winner_counts(lag_all)
    counts_coll = _winner_counts(lag_coll)

    labels     = [f["label"] for f in FEATURES]
    colors     = [f["color"] for f in FEATURES]
    x          = np.arange(len(labels))
    vals_all   = [counts_all.get(l, 0)  for l in labels]
    vals_coll  = [counts_coll.get(l, 0) for l in labels]
    max_n_all  = len(lag_all)
    max_n_coll = len(lag_coll)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Which feature is closest to max acceleration?",
        fontweight="bold"
    )

    for ax, vals, max_n, title_sfx in [
        (axes[0], vals_all,  max_n_all,  f"All sessions  (n={max_n_all})"),
        (axes[1], vals_coll, max_n_coll, f"Collective only  (n={max_n_coll})"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.80, edgecolor="black", lw=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                        str(v), ha="center", va="bottom", fontsize=19)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of sessions  (winner)")
        ax.set_ylim(0, max(vals + [1]) * 1.20)
        ax.set_title(title_sfx, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "timing_winner_counts.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")


# ---------------------------------------------------------------------------
# Plot B — signed lag distributions (violin + scatter)
# ---------------------------------------------------------------------------

def plot_lag_distributions(lag_all: pd.DataFrame, lag_coll: pd.DataFrame,
                           output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.suptitle(
        "Signed lag to max acceleration  (positive = feature after accel peak)",
        fontweight="bold"
    )
    axes[0].axhline(0, color="black", lw=0.9, linestyle="--", alpha=0.5)
    axes[1].axhline(0, color="black", lw=0.9, linestyle="--", alpha=0.5)

    for ax, df_sub, title in [
        (axes[0], lag_all,  f"All sessions  (n={len(lag_all)})"),
        (axes[1], lag_coll, f"Collective only  (n={len(lag_coll)})"),
    ]:
        data_per_feat = []
        positions     = []
        for i, feat in enumerate(FEATURES):
            col_lag = f"lag_{feat['col']}"
            vals = df_sub[col_lag].dropna().values.astype(float)
            data_per_feat.append(vals)
            positions.append(i + 1)

        parts = ax.violinplot(
            [d for d in data_per_feat if len(d) > 1],
            positions=[p for p, d in zip(positions, data_per_feat) if len(d) > 1],
            showmedians=True, showextrema=True,
        )
        # Re-colour violin bodies
        visible_i = [i for i, d in enumerate(data_per_feat) if len(d) > 1]
        for body, vi in zip(parts["bodies"], visible_i):
            body.set_facecolor(FEATURES[vi]["color"])
            body.set_alpha(0.55)
            body.set_edgecolor("black")

        # Scatter overlay
        rng = np.random.default_rng(7)
        for i, (vals, feat) in enumerate(zip(data_per_feat, FEATURES)):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                np.full(len(vals), i + 1) + jitter, vals,
                color=feat["color"], alpha=0.55, s=18, zorder=3,
                edgecolors="black", linewidths=0.4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f["label"] for f in FEATURES])
        ax.set_ylabel("Lag (bins)")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "timing_lag_distributions.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")


# ---------------------------------------------------------------------------
# Plot C — mean absolute lag ± SEM grouped bar
# ---------------------------------------------------------------------------

def plot_mean_abs_lag(lag_all: pd.DataFrame, lag_coll: pd.DataFrame,
                      output_dir: str):
    labels = [f["label"] for f in FEATURES]
    colors = [f["color"] for f in FEATURES]
    x      = np.arange(len(labels))
    width  = 0.35

    def _stats(df_sub, feat):
        col = f"abs_lag_{feat['col']}"
        vals = df_sub[col].dropna().values.astype(float)
        if len(vals) == 0:
            return np.nan, np.nan
        return np.mean(vals), np.std(vals) / np.sqrt(len(vals))

    means_all,  ses_all  = zip(*[_stats(lag_all,  f) for f in FEATURES])
    means_coll, ses_coll = zip(*[_stats(lag_coll, f) for f in FEATURES])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_all  = ax.bar(x - width / 2, means_all,  width,
                       yerr=ses_all,  capsize=5, color=colors, alpha=0.65,
                       edgecolor="black", lw=0.8, label="All sessions")
    bars_coll = ax.bar(x + width / 2, means_coll, width,
                       yerr=ses_coll, capsize=5, color=colors, alpha=0.95,
                       edgecolor="black", lw=1.2, label="Collective only",
                       hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |lag|  (bins)  ± SEM")
    ax.set_title("Mean absolute lag to max acceleration per feature", fontweight="bold")
    ax.legend(frameon=True, framealpha=1.0, loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "timing_mean_abs_lag.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  \u2192 {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse pk_j_summary.csv: which feature best predicts max acceleration?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True,
                   help="Path to pk_j_summary.csv produced by pk_j_analysis.py")
    p.add_argument("--output_dir", default="./notes/peak_timing",
                   help="Folder for output PNGs/SVGs")
    p.add_argument("--stim", type=int, default=None,
                   help="If given, restrict analysis to this stim index only")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    required_cols = {"accel_peak_idx"}
    for feat in FEATURES:
        required_cols.add(feat["col"])
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV is missing columns: {missing}", file=sys.stderr)
        return 1

    if args.stim is not None:
        df = df[df["stim"] == args.stim].copy()
        print(f"  Filtered to stim={args.stim}: {len(df)} rows remaining")
    if df.empty:
        print("No rows to analyse after filtering.", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    lag_all  = _lag_df(df)
    lag_coll = _lag_df(df[df["is_collective"] == True].copy()) if "is_collective" in df.columns else lag_all.iloc[0:0]

    print(f"\nAll sessions: {len(lag_all)} rows with at least one valid feature lag")
    print(f"Collective  : {len(lag_coll)} rows")

    print("\nGenerating plots …")
    plot_closest_counts(lag_all, lag_coll, args.output_dir)
    plot_lag_distributions(lag_all, lag_coll, args.output_dir)
    plot_mean_abs_lag(lag_all, lag_coll, args.output_dir)

    # Print summary table to stdout
    print("\n--- Winner counts (all sessions) ---")
    for label, cnt in _winner_counts(lag_all).items():
        print(f"  {label:<12}: {cnt}")
    print("--- Winner counts (collective only) ---")
    for label, cnt in _winner_counts(lag_coll).items():
        print(f"  {label:<12}: {cnt}")

    print(f"\nDone. Outputs in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
