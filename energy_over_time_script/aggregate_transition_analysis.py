#!/usr/bin/env python3
"""
Aggregate transition-analysis data across multiple sessions.

Scans a root directory tree for session sub-folders that contain the CSV
written by analysis.py (transition_analysis_stim_<N>.csv), and for every
session calls a *hook* function with:

    hook(df, session_dir, session_label, stim_index, output_dir)

where ``df`` is the pandas DataFrame loaded from the file.

A default hook is provided (prints a summary and saves a per-session
energy-vs-time figure with transition points marked).  You can replace it by
passing ``--hook_module`` on the command line, see below.

Expected directory layout produced by process_matlab_file_manyTimes_sbatch.sh:

    <root>/
      <experiment>_rep1/
        begin_reach/   <- transition_analysis_stim_0.csv, …
        mid_reach/
        full_reach/
      <experiment>_rep2/
        begin_reach/
        …

Usage:
    python aggregate_transition_analysis.py <root_dir> \\
        [--stim 0] \\
        [--output_dir OUT] \\
        [--hook_module my_hooks:my_function]

--hook_module accepts "module_path:function_name" (importlib style).
"""

import argparse
import functools
import importlib
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


TRANSITION_CSV_TEMPLATE = "transition_analysis_stim_{stim}.csv"


# ---------------------------------------------------------------------------
# Discovery helpers  (mirrors aggregate_model_quality_plots.py)
# ---------------------------------------------------------------------------

def find_session_dirs(root: Path, stim: int) -> list:
    """Walk root and return every directory that contains the target CSV."""
    target = TRANSITION_CSV_TEMPLATE.format(stim=stim)
    found = []
    for dirpath, _, files in os.walk(root):
        if target in files:
            found.append(Path(dirpath))
    return sorted(found)


def group_by_stim_label(session_dirs: list) -> dict:
    """Group dirs by their last path component (reach-phase / stim label)."""
    groups = defaultdict(list)
    for d in session_dirs:
        groups[d.name].append(d)
    return dict(groups)


def session_label(sdir: Path) -> str:
    """Human-readable label: parent folder name (rep) + leaf."""
    return sdir.parent.name


def load_transition_csv(session_dir: Path, stim: int) -> pd.DataFrame | None:
    p = session_dir / TRANSITION_CSV_TEMPLATE.format(stim=stim)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as exc:
            print(f"  WARNING: could not read {p}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Regression helper
# ---------------------------------------------------------------------------

def _energy_spike_regression(
    df: pd.DataFrame,
    threshold: float,
) -> tuple:
    """
    Build an energy-derivative spike mask and regress it against kinematic vars.

    The spike mask is:
        spike = |Energy_Derivative| > threshold * std(Energy_Derivative)

    For each kinematic variable (position = Kinematics, velocity =
    Kinematics_Derivative, acceleration = np.gradient(Kinematics_Derivative)):

      1. OLS regression   : kinematic_var ~ spike_binary
             slope / R² / p  tell how much the kinematic variable shifts
             when the energy derivative crosses the threshold.
      2. Welch t-test     : mean(kin | spike) vs mean(kin | no-spike)
             Cohen's d quantifies the effect size.
      3. Point-biserial r : equivalent to Pearson between a binary and a
             continuous variable.

    Optionally (if sklearn is available):
      4. Logistic regression: spike ~ kinematic_var
             Coefficient sign shows which direction of kinematic change
             co-occurs with energy spikes; accuracy gives a baseline.

    Returns
    -------
    results_df : pd.DataFrame  — one row per kinematic variable
    spike_mask : np.ndarray bool
    kin_vars   : dict  name → array
    """
    energy_deriv = df["Energy_Derivative"].values
    spike_mask = np.abs(energy_deriv) > threshold * np.std(energy_deriv)
    spike_bin = spike_mask.astype(float)

    kin_pos = df["Kinematics"].values
    kin_vel = df["Kinematics_Derivative"].values
    kin_acc = np.gradient(kin_vel)

    kin_vars = {
        "position": kin_pos,
        "velocity": kin_vel,
        "acceleration": kin_acc,
    }

    # Degenerate mask: no spikes or all spikes — cannot compute any statistic
    _nan_row = dict(
        n_spike=int(spike_mask.sum()), n_no_spike=int((~spike_mask).sum()),
        mean_spike=np.nan, mean_no_spike=np.nan,
        cohens_d=np.nan, t_stat=np.nan, p_ttest=np.nan,
        r_pointbiserial=np.nan, p_pointbiserial=np.nan,
        ols_slope=np.nan, ols_intercept=np.nan, ols_r2=np.nan, ols_p=np.nan,
        logit_coef=np.nan, logit_accuracy=np.nan,
    )
    if spike_mask.sum() < 2 or (~spike_mask).sum() < 2:
        rows = [{"kinematic": name, **_nan_row} for name in kin_vars]
        return pd.DataFrame(rows), spike_mask, kin_vars

    # Optional logistic regression via sklearn
    try:
        from sklearn.linear_model import LogisticRegression as _LR
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    rows = []
    for name, kvar in kin_vars.items():
        k_spike   = kvar[spike_mask]
        k_nospike = kvar[~spike_mask]

        # -- OLS: kvar ~ energy_spike_binary --
        ols_slope, ols_intercept, ols_r, ols_p, _ = stats.linregress(spike_bin, kvar)

        # -- Point-biserial correlation --
        r_pb, p_pb = stats.pointbiserialr(spike_mask, kvar)

        # -- Welch t-test + Cohen's d --
        if len(k_spike) > 1 and len(k_nospike) > 1:
            t_stat, p_ttest = stats.ttest_ind(k_spike, k_nospike, equal_var=False)
            pooled_var = (
                np.var(k_spike, ddof=1) * (len(k_spike) - 1)
                + np.var(k_nospike, ddof=1) * (len(k_nospike) - 1)
            ) / (len(k_spike) + len(k_nospike) - 2)
            cohens_d = (np.mean(k_spike) - np.mean(k_nospike)) / (np.sqrt(pooled_var) + 1e-12)
        else:
            t_stat = p_ttest = cohens_d = np.nan

        # -- Logistic: spike ~ kvar (optional) --
        logit_coef = logit_accuracy = np.nan
        if _has_sklearn and len(np.unique(spike_mask)) > 1:
            try:
                lr = _LR(max_iter=1000, solver="lbfgs")
                lr.fit(kvar.reshape(-1, 1), spike_mask.astype(int))
                logit_coef = float(lr.coef_[0, 0])
                logit_accuracy = float(lr.score(kvar.reshape(-1, 1), spike_mask.astype(int)))
            except Exception:
                pass

        rows.append({
            "kinematic":        name,
            "n_spike":          int(spike_mask.sum()),
            "n_no_spike":       int((~spike_mask).sum()),
            "mean_spike":       float(np.mean(k_spike))   if len(k_spike)   else np.nan,
            "mean_no_spike":    float(np.mean(k_nospike)) if len(k_nospike) else np.nan,
            "cohens_d":         cohens_d,
            "t_stat":           t_stat,
            "p_ttest":          p_ttest,
            "r_pointbiserial":  r_pb,
            "p_pointbiserial":  p_pb,
            "ols_slope":        ols_slope,
            "ols_intercept":    ols_intercept,
            "ols_r2":           ols_r ** 2,
            "ols_p":            ols_p,
            "logit_coef":       logit_coef,
            "logit_accuracy":   logit_accuracy,
        })

    return pd.DataFrame(rows), spike_mask, kin_vars


# ---------------------------------------------------------------------------
# Default hook  — replace this with your own logic
# ---------------------------------------------------------------------------

def default_session_hook(
    df: pd.DataFrame,
    session_dir: Path,
    label: str,
    stim: int,
    output_dir: Path,
    threshold: float = 1.0,
) -> None:
    """
    Default hook called once per session.

    For each session:
      1. Computes the energy-derivative spike mask:
             |Energy_Derivative| > threshold * std(Energy_Derivative)
      2. Runs OLS, t-test/Cohen's d, point-biserial correlation, and
         (if sklearn is available) logistic regression against position,
         velocity, and acceleration derived from the Kinematics columns.
      3. Saves a 4-row figure:
             Row 0 : Energy derivative + threshold bands + spike markers
             Rows 1-3: Each kinematic variable time-series coloured by
                       spike / no-spike, with box-plot inset.
      4. Saves a CSV of regression results.

    Parameters
    ----------
    df          : DataFrame from transition_analysis_stim_<stim>.csv
    session_dir : Path to the session directory
    label       : Human-readable session label
    stim        : Stimulus index
    output_dir  : Where to write outputs
    threshold   : Std-multiplier for the spike threshold (default 1.0)
    """
    t = df["Time_Index"].values if "Time_Index" in df.columns else np.arange(len(df))
    energy_deriv = df["Energy_Derivative"].values

    results_df, spike_mask, kin_vars = _energy_spike_regression(df, threshold)

    # ---- print summary ----
    print(f"    [{label}]  n_spike={spike_mask.sum()}  "
          f"({100*spike_mask.mean():.1f}% of {len(df)} time-points)  "
          f"threshold={threshold}×σ")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 120):
        print(results_df.to_string(index=False))

    # ---- figure ----
    kin_names  = list(kin_vars.keys())
    kin_arrays = list(kin_vars.values())
    n_rows = 1 + len(kin_names)  # energy row + one per kinematic var

    fig = plt.figure(figsize=(14, 3.2 * n_rows))
    # left column: time series; right column: box-plot comparison
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[4, 1], hspace=0.45, wspace=0.25)

    spike_color   = "#d62728"   # red
    nospike_color = "#1f77b4"   # blue

    # -- Row 0: Energy derivative --
    ax0 = fig.add_subplot(gs[0, :])
    thresh_val = threshold * np.std(energy_deriv)
    ax0.plot(t, energy_deriv, lw=0.7, color="dimgray", label="Energy deriv.")
    ax0.axhline( thresh_val, color=spike_color, lw=1.2, ls="--",
                 label=f"+{threshold}σ threshold")
    ax0.axhline(-thresh_val, color=spike_color, lw=1.2, ls="--",
                 label=f"−{threshold}σ threshold")
    ax0.fill_between(t, energy_deriv, where=spike_mask,
                     color=spike_color, alpha=0.25, label="spike region")
    ax0.set_ylabel("Energy deriv.")
    ax0.set_xlabel("Time index")
    ax0.set_title(f"{label}  |  stim {stim}  |  Energy-derivative spikes "
                  f"(threshold = {threshold}σ)")
    ax0.legend(fontsize=7, ncol=4, loc="upper right")
    ax0.grid(alpha=0.25)

    # -- Rows 1-3: kinematic variables --
    row_labels = ["(a)", "(b)", "(c)"]
    for i, (name, kvar) in enumerate(zip(kin_names, kin_arrays), start=1):
        row = results_df[results_df["kinematic"] == name].iloc[0]

        # Time series
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_ts.plot(t[~spike_mask], kvar[~spike_mask],
                   ".", ms=2, color=nospike_color, alpha=0.5, label="no spike")
        ax_ts.plot(t[spike_mask], kvar[spike_mask],
                   ".", ms=3, color=spike_color, alpha=0.7, label="spike")
        # OLS regression line
        x_line = np.array([0.0, 1.0])
        y_line = row["ols_slope"] * x_line + row["ols_intercept"]
        mean_nospike = row["mean_no_spike"]
        mean_spike   = row["mean_spike"]
        ax_ts.axhline(mean_nospike, color=nospike_color, lw=1.2, ls="--", alpha=0.7)
        ax_ts.axhline(mean_spike,   color=spike_color,   lw=1.2, ls="--", alpha=0.7)

        sig_star = (
            "***" if row["p_ttest"] < 0.001 else
            "**"  if row["p_ttest"] < 0.01  else
            "*"   if row["p_ttest"] < 0.05  else "n.s."
        )
        ax_ts.set_ylabel(name.capitalize())
        ax_ts.set_xlabel("Time index")
        ax_ts.set_title(
            f"{row_labels[i-1]}  {name.capitalize()}  |  "
            f"OLS slope={row['ols_slope']:.4f}  R²={row['ols_r2']:.4f}  p={row['ols_p']:.3g}  |  "
            f"Cohen's d={row['cohens_d']:.3f}  {sig_star}  |  "
            f"r_pb={row['r_pointbiserial']:.3f}",
            fontsize=8,
        )
        ax_ts.legend(fontsize=7, markerscale=2)
        ax_ts.grid(alpha=0.25)

        # Box plot
        ax_bx = fig.add_subplot(gs[i, 1])
        bp = ax_bx.boxplot(
            [kvar[~spike_mask], kvar[spike_mask]],
            tick_labels=["no spike", "spike"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", lw=1.5),
        )
        bp["boxes"][0].set_facecolor(nospike_color)
        bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor(spike_color)
        bp["boxes"][1].set_alpha(0.55)
        ax_bx.set_ylabel(name.capitalize())
        ax_bx.set_title(f"d={row['cohens_d']:.2f}  {sig_star}", fontsize=8)
        ax_bx.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Energy-derivative spike → kinematic regression\n"
        f"{label}  |  stim {stim}  |  threshold = {threshold}σ",
        fontsize=11, fontweight="bold",
    )

    out_png = output_dir / f"energy_spike_regression_{label}_stim_{stim}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved figure : {out_png}")

    # ---- save regression CSV ----
    results_df.insert(0, "session", label)
    results_df.insert(1, "stim",    stim)
    results_df.insert(2, "threshold", threshold)
    out_csv = output_dir / f"energy_spike_regression_{label}_stim_{stim}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"    Saved results: {out_csv}")


# ---------------------------------------------------------------------------
# Hook loader
# ---------------------------------------------------------------------------

def load_hook(spec: str | None):
    """
    Load a hook callable from "module_path:function_name" string.
    Falls back to the built-in default_session_hook when spec is None.
    """
    if spec is None:
        return default_session_hook

    if ":" not in spec:
        raise ValueError(
            f"--hook_module must be 'module_path:function_name', got: {spec!r}"
        )
    mod_path, func_name = spec.rsplit(":", 1)

    # Allow plain file paths (e.g. ./my_hooks.py)
    mod_file = Path(mod_path)
    if mod_file.suffix == ".py" and mod_file.exists():
        sys.path.insert(0, str(mod_file.parent))
        mod_path = mod_file.stem

    module = importlib.import_module(mod_path)
    hook = getattr(module, func_name)
    if not callable(hook):
        raise TypeError(f"{func_name!r} in module {mod_path!r} is not callable")
    return hook


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(root: Path, stim: int, output_dir: Path, hook) -> None:
    session_dirs = find_session_dirs(root, stim)
    if not session_dirs:
        target = TRANSITION_CSV_TEMPLATE.format(stim=stim)
        print(f"No session directories containing '{target}' found under:\n  {root}")
        return

    groups = group_by_stim_label(session_dirs)
    print(
        f"Found {len(session_dirs)} session folder(s) across "
        f"{len(groups)} reach-phase group(s):"
    )
    for phase, dirs in sorted(groups.items()):
        print(f"  [{phase}]  {len(dirs)} session(s)")

    print()
    for sdir in session_dirs:
        label = session_label(sdir)
        df = load_transition_csv(sdir, stim)
        if df is None:
            print(f"  SKIP {sdir}  (could not load CSV)")
            continue

        print(f"  Calling hook for session: {label}  ({sdir})")
        hook(df, sdir, label, stim, output_dir)


# ---------------------------------------------------------------------------
# Cross-session meta-analysis
# ---------------------------------------------------------------------------

def _cohens_d_se(d: float, n1: int, n2: int) -> float:
    """Standard error of Cohen's d (Hedges & Olkin approximation)."""
    n = n1 + n2
    return float(np.sqrt(n / (n1 * n2) + d ** 2 / (2 * (n - 2))))


def meta_analysis(output_dir: Path, stim: int) -> None:
    """
    Pool all per-session energy_spike_regression CSVs written by the default
    hook and run a cross-session meta-analysis for each kinematic variable:

      1. **Sign test** (binomial): is the direction of Cohen's d consistent?
      2. **Fisher's combined p-value**: is there a global effect, even if
         individual sessions are underpowered?
      3. **Inverse-variance-weighted pooled Cohen's d** with 95 % CI.
      4. **Forest plot**: one column per kinematic variable, one row per
         session, with a pooled-effect diamond at the bottom.

    Results are saved as:
      aggregate_meta_analysis_stim_<N>.csv
      aggregate_meta_analysis_stim_<N>.png
    """
    pattern = f"energy_spike_regression_*_stim_{stim}.csv"
    csv_files = sorted(output_dir.glob(pattern))
    if not csv_files:
        print(f"  Meta-analysis: no files matching '{pattern}' in {output_dir}")
        return

    all_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    kin_names = all_df["kinematic"].unique().tolist()

    # ---- per-variable meta stats ----
    meta_rows = []
    forest_data = {}   # name → list of (session, d, lo, hi)

    for name in kin_names:
        sub = all_df[all_df["kinematic"] == name].copy()
        sessions = sub["session"].tolist()
        d_vals   = sub["cohens_d"].values
        n1_vals  = sub["n_spike"].values.astype(int)
        n2_vals  = sub["n_no_spike"].values.astype(int)
        p_ttest  = sub["p_ttest"].values

        # -- per-session SE and CI (skip degenerate sessions) --
        se_vals = np.array([
            _cohens_d_se(d, n1, n2) if (n1 >= 2 and n2 >= 2 and np.isfinite(d)) else np.nan
            for d, n1, n2 in zip(d_vals, n1_vals, n2_vals)
        ])
        ci_lo = d_vals - 1.96 * se_vals
        ci_hi = d_vals + 1.96 * se_vals

        forest_data[name] = list(zip(sessions, d_vals, ci_lo, ci_hi))

        # -- sign test: how many d < 0 (negative = spike → lower kin) --
        n_neg    = int((d_vals < 0).sum())
        n_pos    = int((d_vals > 0).sum())
        n_nonzero = n_neg + n_pos
        p_sign   = stats.binomtest(n_neg, n_nonzero, 0.5).pvalue if n_nonzero > 0 else np.nan

        # -- Fisher's combined p-value --
        p_clean = p_ttest[np.isfinite(p_ttest) & (p_ttest > 0)]
        if len(p_clean) > 0:
            fisher_stat = -2.0 * np.sum(np.log(p_clean))
            fisher_p = stats.chi2.sf(fisher_stat, df=2 * len(p_clean))
        else:
            fisher_stat = fisher_p = np.nan

        # -- inverse-variance-weighted pooled d --
        var_vals = se_vals ** 2
        weights  = 1.0 / np.where(var_vals > 0, var_vals, np.nan)
        finite   = np.isfinite(weights)
        if finite.any():
            w       = weights[finite]
            d_f     = d_vals[finite]
            pooled_d  = np.sum(w * d_f) / np.sum(w)
            pooled_se = np.sqrt(1.0 / np.sum(w))
            pooled_lo = pooled_d - 1.96 * pooled_se
            pooled_hi = pooled_d + 1.96 * pooled_se
            pooled_z  = pooled_d / pooled_se
            pooled_p  = 2.0 * stats.norm.sf(abs(pooled_z))
        else:
            pooled_d = pooled_se = pooled_lo = pooled_hi = pooled_z = pooled_p = np.nan

        meta_rows.append({
            "kinematic":      name,
            "n_sessions":     len(sub),
            "n_d_negative":   n_neg,
            "n_d_positive":   n_pos,
            "p_sign_test":    p_sign,
            "fisher_stat":    fisher_stat,
            "fisher_p":       fisher_p,
            "pooled_cohens_d":pooled_d,
            "pooled_se":      pooled_se,
            "pooled_ci_lo":   pooled_lo,
            "pooled_ci_hi":   pooled_hi,
            "pooled_z":       pooled_z,
            "pooled_p":       pooled_p,
        })

    meta_df = pd.DataFrame(meta_rows)

    # ---- print ----
    print("\n" + "=" * 72)
    print(f"  META-ANALYSIS  (stim {stim},  {len(csv_files)} sessions)")
    print("=" * 72)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 120):
        print(meta_df.to_string(index=False))
    print()
    for _, row in meta_df.iterrows():
        sign_str   = _sig_star(row["p_sign_test"])
        fisher_str = _sig_star(row["fisher_p"])
        pool_str   = _sig_star(row["pooled_p"])
        direction  = "↓ lower during spike" if row["pooled_cohens_d"] < 0 else "↑ higher during spike"
        print(
            f"  {row['kinematic']:14s}  "
            f"sign-test p={row['p_sign_test']:.3f}{sign_str}  "
            f"Fisher p={row['fisher_p']:.3f}{fisher_str}  "
            f"pooled d={row['pooled_cohens_d']:+.3f} [{row['pooled_ci_lo']:+.3f}, {row['pooled_ci_hi']:+.3f}] "
            f"p={row['pooled_p']:.3f}{pool_str}  {direction}"
        )
    print()

    # ---- forest plot ----
    n_kin  = len(kin_names)
    n_sess = max(len(v) for v in forest_data.values())

    fig, axes = plt.subplots(1, n_kin, figsize=(5.5 * n_kin, max(6, 0.45 * n_sess + 3)),
                             sharey=False)
    if n_kin == 1:
        axes = [axes]

    for ax, name in zip(axes, kin_names):
        entries = forest_data[name]          # (session, d, lo, hi)
        pool_row = meta_df[meta_df["kinematic"] == name].iloc[0]

        sessions_plot = [e[0] for e in entries]
        d_plot   = np.array([e[1] for e in entries])
        lo_plot  = np.array([e[2] for e in entries])
        hi_plot  = np.array([e[3] for e in entries])

        # sort by d for readability
        order = np.argsort(d_plot)
        sessions_plot = [sessions_plot[i] for i in order]
        d_plot  = d_plot[order]
        lo_plot = lo_plot[order]
        hi_plot = hi_plot[order]

        y = np.arange(len(sessions_plot))

        # per-session CIs
        colors = ["#d62728" if d < 0 else "#1f77b4" for d in d_plot]
        ax.barh(y, d_plot, xerr=np.stack([d_plot - lo_plot, hi_plot - d_plot]),
                color=colors, alpha=0.65, height=0.55,
                error_kw=dict(ecolor="black", lw=0.9, capsize=3))

        # reference line
        ax.axvline(0, color="black", lw=1.0, ls="--", zorder=5)

        # pooled diamond
        pd_  = pool_row["pooled_cohens_d"]
        plo_ = pool_row["pooled_ci_lo"]
        phi_ = pool_row["pooled_ci_hi"]
        p_p  = pool_row["pooled_p"]
        sign = pool_row["p_sign_test"]
        y_diam = -1.4
        half_h = 0.5
        diamond_x = [plo_, pd_, phi_, pd_, plo_]
        diamond_y = [y_diam, y_diam + half_h, y_diam, y_diam - half_h, y_diam]
        diam_color = "#d62728" if pd_ < 0 else "#1f77b4"
        ax.fill(diamond_x, diamond_y, color=diam_color, alpha=0.85, zorder=6)
        ax.plot(diamond_x, diamond_y, color="black", lw=0.8, zorder=7)
        ax.text(pd_, y_diam - 0.85,
                f"d={pd_:+.3f}\np={p_p:.3f}{_sig_star(p_p)}\nsign p={sign:.3f}{_sig_star(sign)}",
                ha="center", va="top", fontsize=7, zorder=8)

        ax.set_yticks(y)
        ax.set_yticklabels(sessions_plot, fontsize=7)
        ax.set_ylim(y_diam - 2.0, len(y) - 0.5)
        ax.set_xlabel("Cohen's d  (spike vs no-spike)")
        ax.set_title(f"{name.capitalize()}", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # column header annotations
        ax.text(0.02, 1.01,
                f"sign-test: {_sig_star(sign) or 'n.s.'}  "
                f"Fisher: {_sig_star(pool_row['fisher_p']) or 'n.s.'}",
                transform=ax.transAxes, fontsize=7, va="bottom")

    fig.suptitle(
        f"Cross-session meta-analysis — energy-derivative spikes → kinematics\n"
        f"stim {stim}  |  {len(csv_files)} sessions  |  "
        f"red = spike period lower, blue = spike period higher",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out_png = output_dir / f"aggregate_meta_analysis_stim_{stim}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved forest plot : {out_png}")

    out_csv = output_dir / f"aggregate_meta_analysis_stim_{stim}.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"  Saved meta CSV    : {out_csv}")


def _sig_star(p) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Threshold sensitivity sweep
# ---------------------------------------------------------------------------

def _pool_session_results(session_results: list) -> dict:
    """
    Given a list of per-session result DataFrames (one per session, as returned
    by _energy_spike_regression), compute IVW-pooled Cohen's d, CI, and p for
    each kinematic variable.

    Returns a dict: kinematic_name → {pooled_d, pooled_se, ci_lo, ci_hi, pooled_p,
                                       n_neg, n_pos, p_sign, fisher_p}
    """
    all_df = pd.concat(session_results, ignore_index=True)
    out = {}
    for name in all_df["kinematic"].unique():
        sub     = all_df[all_df["kinematic"] == name]
        d_vals  = sub["cohens_d"].values
        n1_vals = sub["n_spike"].values.astype(int)
        n2_vals = sub["n_no_spike"].values.astype(int)
        p_ttest = sub["p_ttest"].values

        se_vals = np.array([
            _cohens_d_se(d, n1, n2) if (n1 >= 2 and n2 >= 2 and np.isfinite(d)) else np.nan
            for d, n1, n2 in zip(d_vals, n1_vals, n2_vals)
        ])

        # IVW pooled d
        var_vals = se_vals ** 2
        weights  = 1.0 / np.where(var_vals > 0, var_vals, np.nan)
        finite   = np.isfinite(weights)
        if finite.any():
            w         = weights[finite]
            d_f       = d_vals[finite]
            pooled_d  = np.sum(w * d_f) / np.sum(w)
            pooled_se = np.sqrt(1.0 / np.sum(w))
            pooled_z  = pooled_d / pooled_se
            pooled_p  = 2.0 * stats.norm.sf(abs(pooled_z))
        else:
            pooled_d = pooled_se = pooled_z = pooled_p = np.nan

        # sign test
        n_neg    = int((d_vals < 0).sum())
        n_pos    = int((d_vals > 0).sum())
        n_nz     = n_neg + n_pos
        p_sign   = stats.binomtest(n_neg, n_nz, 0.5).pvalue if n_nz > 0 else np.nan

        # Fisher
        p_clean = p_ttest[np.isfinite(p_ttest) & (p_ttest > 0)]
        if len(p_clean):
            fisher_p = stats.chi2.sf(-2.0 * np.sum(np.log(p_clean)), df=2 * len(p_clean))
        else:
            fisher_p = np.nan

        out[name] = dict(
            pooled_d=pooled_d, pooled_se=pooled_se,
            ci_lo=pooled_d - 1.96 * pooled_se if np.isfinite(pooled_se) else np.nan,
            ci_hi=pooled_d + 1.96 * pooled_se if np.isfinite(pooled_se) else np.nan,
            pooled_p=pooled_p, p_sign=p_sign, fisher_p=fisher_p,
            n_neg=n_neg, n_pos=n_pos,
        )
    return out


def threshold_sweep(
    root: Path,
    stim: int,
    output_dir: Path,
    thresholds: list,
) -> None:
    """
    Re-run the energy-spike regression at each value in ``thresholds`` across
    all sessions and plot the IVW-pooled Cohen's d (with 95 % CI) vs threshold
    for each kinematic variable.

    This is the "dose-response" check: if the signal is real, the pooled
    effect size should strengthen (more negative for acceleration) as the
    threshold rises and only the largest / most genuine energy events are kept.

    Outputs
    -------
    threshold_sweep_stim_<N>.csv   — pooled stats at every threshold × kinematic
    threshold_sweep_stim_<N>.png   — sensitivity curve figure
    """
    session_dirs = find_session_dirs(root, stim)
    if not session_dirs:
        print(f"  Threshold sweep: no sessions found under {root}")
        return

    # Pre-load all session DataFrames once
    session_dfs = {}
    for sdir in session_dirs:
        lab = session_label(sdir)
        df  = load_transition_csv(sdir, stim)
        if df is not None:
            session_dfs[lab] = df

    if not session_dfs:
        print("  Threshold sweep: could not load any session CSVs.")
        return

    n_sess = len(session_dfs)
    print(f"\n  Threshold sweep: {n_sess} sessions × {len(thresholds)} thresholds ...")

    sweep_rows = []
    # pooled_by_thresh: thresh → kinematic → pooled stats dict
    pooled_by_thresh = {}

    for thr in thresholds:
        session_results = []
        for lab, df in session_dfs.items():
            res_df, _, _ = _energy_spike_regression(df, thr)
            res_df.insert(0, "session", lab)
            session_results.append(res_df)

        pooled = _pool_session_results(session_results)
        pooled_by_thresh[thr] = pooled

        for kin, stats_d in pooled.items():
            sweep_rows.append({"threshold": thr, "kinematic": kin, **stats_d})

        # brief console summary
        parts = [f"thr={thr:.1f}σ"]
        for kin, sd in pooled.items():
            parts.append(
                f"  {kin}: d={sd['pooled_d']:+.3f} "
                f"[{sd['ci_lo']:+.3f},{sd['ci_hi']:+.3f}]"
                f" p={sd['pooled_p']:.3f}{_sig_star(sd['pooled_p'])}"
            )
        print("    " + "  |".join(parts))

    sweep_df = pd.DataFrame(sweep_rows)
    out_csv  = output_dir / f"threshold_sweep_stim_{stim}.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"  Saved sweep CSV : {out_csv}")

    # ---- figure ----
    kin_names = sweep_df["kinematic"].unique().tolist()
    thr_arr   = np.array(thresholds)

    # colour palette: one colour per kinematic variable
    kin_colors = {"position": "#1f77b4", "velocity": "#ff7f0e", "acceleration": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    axes_map  = {name: ax for name, ax in zip(["pooled_d", "pooled_p", "p_sign"], axes)}

    # ---- Panel 1: pooled Cohen's d vs threshold ----
    ax_d = axes[0]
    for name in kin_names:
        sub   = sweep_df[sweep_df["kinematic"] == name]
        d_arr = sub["pooled_d"].values
        lo    = sub["ci_lo"].values
        hi    = sub["ci_hi"].values
        col   = kin_colors.get(name, "gray")
        ax_d.plot(thr_arr, d_arr, "o-", color=col, lw=2.0, ms=6, label=name)
        ax_d.fill_between(thr_arr, lo, hi, color=col, alpha=0.15)
        # significance markers above/below each point
        for tx, dy, py in zip(thr_arr, d_arr, sub["pooled_p"].values):
            star = _sig_star(py)
            if star:
                offset = -0.008 if dy < 0 else 0.008
                ax_d.text(tx, dy + offset, star, ha="center",
                          va="top" if dy < 0 else "bottom",
                          fontsize=9, color=col, fontweight="bold")

    ax_d.axhline(0, color="black", lw=1.0, ls="--")
    ax_d.set_xlabel("Threshold (× σ of Energy Derivative)")
    ax_d.set_ylabel("Pooled Cohen's d  (IVW)")
    ax_d.set_title("Effect size vs threshold\n(CI shaded; * p<.05, ** p<.01, *** p<.001)",
                   fontsize=9)
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3)
    ax_d.set_xticks(thr_arr)

    # ---- Panel 2: pooled p-value (log scale) vs threshold ----
    ax_p = axes[1]
    for name in kin_names:
        sub  = sweep_df[sweep_df["kinematic"] == name]
        col  = kin_colors.get(name, "gray")
        p_arr = np.clip(sub["pooled_p"].values, 1e-6, 1.0)
        ax_p.semilogy(thr_arr, p_arr, "o-", color=col, lw=2.0, ms=6, label=name)

    ax_p.axhline(0.05, color="black", lw=1.0, ls="--", label="p=0.05")
    ax_p.axhline(0.01, color="black", lw=0.7, ls=":",  label="p=0.01")
    ax_p.set_xlabel("Threshold (× σ of Energy Derivative)")
    ax_p.set_ylabel("Pooled p-value  (log scale)")
    ax_p.set_title("Significance vs threshold\n(lower = more significant)", fontsize=9)
    ax_p.legend(fontsize=8)
    ax_p.grid(alpha=0.3, which="both")
    ax_p.set_xticks(thr_arr)

    # ---- Panel 3: fraction of sessions going negative ----
    ax_s = axes[2]
    for name in kin_names:
        sub   = sweep_df[sweep_df["kinematic"] == name]
        col   = kin_colors.get(name, "gray")
        frac  = sub["n_neg"].values / (sub["n_neg"].values + sub["n_pos"].values)
        ax_s.plot(thr_arr, frac, "o-", color=col, lw=2.0, ms=6, label=name)
        # sign-test p markers
        for tx, fx, sp in zip(thr_arr, frac, sub["p_sign"].values):
            star = _sig_star(sp)
            if star:
                ax_s.text(tx, fx + 0.01, star, ha="center", va="bottom",
                          fontsize=9, color=col, fontweight="bold")

    ax_s.axhline(0.5, color="black", lw=1.0, ls="--", label="chance (0.5)")
    ax_s.set_xlabel("Threshold (× σ of Energy Derivative)")
    ax_s.set_ylabel("Fraction of sessions with d < 0")
    ax_s.set_title("Direction consistency vs threshold\n(> 0.5 = mostly negative effect)",
                   fontsize=9)
    ax_s.set_ylim(0, 1)
    ax_s.legend(fontsize=8)
    ax_s.grid(alpha=0.3)
    ax_s.set_xticks(thr_arr)

    fig.suptitle(
        f"Threshold sensitivity sweep — energy-derivative spikes → kinematics\n"
        f"stim {stim}  |  {n_sess} sessions  |  "
        f"thresholds: {thresholds}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    out_png = output_dir / f"threshold_sweep_stim_{stim}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved sweep plot: {out_png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        help="Root folder containing session sub-folders with transition analysis CSVs",
    )
    parser.add_argument(
        "--stim", type=int, default=0,
        help="Stimulus index used to select transition_analysis_stim_<N>.csv (default: 0)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save output files (default: <root_dir>/aggregate_transition)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help=(
            "Std-multiplier for the energy-derivative spike threshold: "
            "|Energy_Derivative| > threshold × σ  (default: 1.0).  "
            "Only used by the built-in default hook."
        ),
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=(
            "Run a threshold sensitivity sweep after the per-session analysis.  "
            "Tests a range of thresholds and plots pooled Cohen's d vs threshold."
        ),
    )
    parser.add_argument(
        "--sweep_thresholds", default="0.5,1.0,1.5,2.0,2.5,3.0",
        metavar="T1,T2,...",
        help=(
            "Comma-separated threshold values for the sensitivity sweep "
            "(default: 0.5,1.0,1.5,2.0,2.5,3.0).  Requires --sweep."
        ),
    )
    parser.add_argument(
        "--hook_module", default=None,
        metavar="MODULE:FUNCTION",
        help=(
            "Custom hook callable in 'module_path:function_name' format.  "
            "The function must accept (df, session_dir, label, stim, output_dir).  "
            "Example: my_hooks:plot_custom"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else root / "aggregate_transition"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hook_module is None:
        hook = functools.partial(default_session_hook, threshold=args.threshold)
    else:
        hook = load_hook(args.hook_module)
    hook_name = getattr(hook, "__qualname__", None) or getattr(hook.func, "__qualname__", repr(hook))
    print(f"Hook: {hook_name}")
    print(f"Root: {root}")
    print(f"Stim: {args.stim}")
    print(f"Output dir: {output_dir}\n")

    run(root, args.stim, output_dir, hook)

    if args.hook_module is None:
        print("\nRunning cross-session meta-analysis ...")
        meta_analysis(output_dir, args.stim)

    if args.sweep and args.hook_module is None:
        thresholds = [float(t) for t in args.sweep_thresholds.split(",")]
        threshold_sweep(root, args.stim, output_dir, thresholds)

    print(f"\nDone.  Output in: {output_dir}")


if __name__ == "__main__":
    main()
