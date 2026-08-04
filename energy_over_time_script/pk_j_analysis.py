#!/usr/bin/env python3
# coding: utf-8
"""
pk_j_analysis.py
----------------
For each session/stim, test whether a J coupling peak co-occurs with a
collective state — defined by the Ising model fitting the data better than
the independent model AND the two models diverging meaningfully.

Hypothesis
----------
A J peak occurs when:
    1. r_ising >= r_independent − 0.02   (Ising fits at least as well as indep.)
    2. ising_indep_dist > 0.10           (the two models differ meaningfully)

Both conditions together flag a session as "collective".

Usage
-----
  python pk_j_analysis.py \\
      --data_folder /path/to/outputs \\
      --window 350 475 \\
      --output_dir ./notes/pk_analysis
"""

import argparse
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (mannwhitneyu, shapiro, wilcoxon)

from src.util import find_file_recursive
from src.peak_detection import (
    detect_j_peak          as _j_peak_detect,
    detect_signal_extremum as _signal_extremum_detect,
    gaussian_smooth        as _gaussian_smooth,
)

plt.rcParams.update({
    "axes.grid":        False,
    "font.size":        17,
    "axes.titlesize":   19,
    "axes.labelsize":   17,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  16,
    "figure.titlesize": 19,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: str, **kwargs):
    """Save *fig* as PNG and SVG (svg.fonttype='none' for Affinity Designer)."""
    fig.savefig(path, **kwargs)
    svg_path = os.path.splitext(path)[0] + ".svg"
    svg_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg_path, **svg_kwargs)


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _na(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------

def _mean_ts(df: pd.DataFrame, col: str) -> np.ndarray:
    parts = []
    for _, grp in df.groupby("reach_idx"):
        parts.append(grp[col].values)
    if not parts:
        return np.array([])
    n = min(len(p) for p in parts)
    return np.mean([p[:n] for p in parts], axis=0)


def _velocity_ts(df: pd.DataFrame) -> np.ndarray:
    parts = []
    for _, grp in df.groupby("reach_idx"):
        x = grp["x"].values
        parts.append(np.diff(x, prepend=x[0]))
    if not parts:
        return np.array([])
    n = min(len(p) for p in parts)
    return np.mean([p[:n] for p in parts], axis=0)


def _acceleration_ts(df: pd.DataFrame) -> np.ndarray:
    v = _velocity_ts(df)
    return np.diff(v, prepend=v[0]) if len(v) else np.array([])


# ---------------------------------------------------------------------------
# P(K) reader & metrics
# ---------------------------------------------------------------------------

def _load_pk(stim_dir: str):
    path = os.path.join(stim_dir, "model_quality_summary_P_K.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        required = {"K", "P_data", "P_ising", "P_independent"}
        if not required.issubset(df.columns):
            return None
        return df.dropna(subset=["P_data", "P_ising", "P_independent"])
    except Exception:
        return None


def _pk_metrics(pk: pd.DataFrame) -> dict:
    p_data  = pk["P_data"].values.astype(float)
    p_ising = pk["P_ising"].values.astype(float)
    p_indep = pk["P_independent"].values.astype(float)

    def _safe_r(a, b):
        if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    r_ising          = _safe_r(p_data, p_ising)
    r_independent    = _safe_r(p_data, p_indep)
    r_ising_vs_indep = _safe_r(p_ising, p_indep)
    ising_indep_dist = float(np.sum(np.abs(p_ising - p_indep)))
    ising_closer     = bool(
        (not np.isnan(r_ising)) and (not np.isnan(r_independent))
        and r_ising >= r_independent - 0.02
    )
    collective_score = ising_indep_dist * (1.0 if ising_closer else -1.0)

    return {
        "r_ising":          r_ising,
        "r_independent":    r_independent,
        "r_ising_vs_indep": r_ising_vs_indep,
        "ising_indep_dist": ising_indep_dist,
        "ising_closer":     ising_closer,
        "collective_score": collective_score,
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _stim_from_path(path: str):
    for part in path.replace("\\", "/").split("/"):
        m = re.fullmatch(r"stim_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def _session_id_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    stim_idx = next(
        (i for i, p in enumerate(parts) if re.fullmatch(r"stim_\d+", p)), None)
    search_parts = parts[:stim_idx] if stim_idx is not None else parts
    for part in reversed(search_parts):
        m = re.search(r"(\d{6})", part)
        if m:
            return m.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Per-session 6-panel plot
# ---------------------------------------------------------------------------

def _plot_session(session_id: str, stim: int,
                  vel_ts, accel_ts, fr_ts, energy_ts, j_ts,
                  pk_df, pk_mets: dict,
                  w_lo: int, w_hi: int,
                  vel_peak_idx: int, accel_peak_idx: int,
                  has_j_peak: bool, j_peak_idx,
                  j_peak_ratio: float, timing: str,
                  save_path: str,
                  smooth_sigma: float = 5.0,
                  h_ts=None, h_peak_idx=None,
                  fr_peak_idx=None, fr_trough_idx=None):
    """
    6-panel figure per session:
      0: X velocity
      1: X acceleration
      2: Firing rate   + mean line + peak/trough markers
      3: Ising energy
      4: J coupling + H field (twin y-axis)
      5: P(K) data vs Ising vs independent + hypothesis annotation
    """
    is_collective = (
        pk_mets["ising_closer"] and pk_mets["ising_indep_dist"] > 0.1
        if pk_mets else False
    )
    hypothesis_met = has_j_peak and is_collective

    title = (
        f"Session {session_id}  |  Stim {stim}  |  J {timing.upper()}\n"
        f"J peak: {'YES' if has_j_peak else 'NO'}  "
        f"{'idx=' + str(j_peak_idx) + '  ' if has_j_peak and j_peak_idx is not None else ''}"
        f"ratio={_na(j_peak_ratio)}\n"
    )
    if pk_mets:
        title += (
            f"P(K): r_ising={_na(pk_mets['r_ising'])}  "
            f"r_indep={_na(pk_mets['r_independent'])}  "
            f"Ising_indep_dist={_na(pk_mets['ising_indep_dist'])}\n"
            f"IsingCloser(\u00b10.02)={pk_mets['ising_closer']}  "
            f"Hypothesis={'MET \u2713' if hypothesis_met else 'not met'}"
        )

    fig, axes = plt.subplots(6, 1, figsize=(14, 30), sharex=False)
    fig.suptitle(title, fontsize=17, fontweight="bold")

    def _shade(ax):
        ax.axvspan(w_lo, w_hi, color="gold", alpha=0.12, zorder=0)
        ax.axvline(w_lo, color="goldenrod", linestyle="--", lw=0.9, alpha=0.5)
        ax.axvline(w_hi, color="goldenrod", linestyle="--", lw=0.9, alpha=0.5)
        ax.axvline(accel_peak_idx, color="green", linestyle="--", lw=1.4, alpha=0.6,
                   label="Accel peak")
        ax.axvline(vel_peak_idx,   color="deepskyblue", linestyle="--", lw=1.4, alpha=0.6,
                   label="Vel peak")

    # ── Panel 0: Velocity ────────────────────────────────────────────────
    axes[0].plot(vel_ts, color="navy", lw=1.5, label="X velocity")
    _shade(axes[0])
    axes[0].set_ylabel("X Velocity")
    axes[0].legend(fontsize=16, loc="upper right", frameon=True, framealpha=1.0)

    # ── Panel 1: Acceleration ────────────────────────────────────────────
    axes[1].plot(accel_ts, color="steelblue", lw=1.5, label="X acceleration")
    _shade(axes[1])
    axes[1].set_ylabel("X Acceleration")
    axes[1].legend(fontsize=16, loc="lower right", frameon=True, framealpha=1.0)

    # ── Panel 2: Firing rate ─────────────────────────────────────────────
    axes[2].plot(fr_ts, color="darkgreen", lw=1.5, label="Firing rate")
    fr_mean = np.nanmean(fr_ts) if len(fr_ts) > 0 else np.nan
    axes[2].axhline(fr_mean, color="darkgreen", linestyle="--",
                    lw=1.2, alpha=0.5, label="_nolegend_")
    _shade(axes[2])
    if fr_peak_idx is not None and not (isinstance(fr_peak_idx, float) and np.isnan(fr_peak_idx)):
        idx = int(fr_peak_idx)
        axes[2].axvline(idx, color="limegreen", linestyle=":", lw=1.8, alpha=0.9)
        axes[2].plot(idx, fr_ts[idx], "^", color="limegreen", markersize=10,
                     zorder=5, label=f"FR peak (idx={idx})")
    if fr_trough_idx is not None and not (isinstance(fr_trough_idx, float) and np.isnan(fr_trough_idx)):
        idx = int(fr_trough_idx)
        axes[2].axvline(idx, color="darkred", linestyle=":", lw=1.8, alpha=0.9)
        axes[2].plot(idx, fr_ts[idx], "v", color="darkred", markersize=10,
                     zorder=5, label=f"FR trough (idx={idx})")
    axes[2].set_ylabel("Firing Rate")
    axes[2].legend(fontsize=16, loc="lower right", frameon=True, framealpha=1.0)

    # ── Panel 3: Energy ──────────────────────────────────────────────────
    axes[3].plot(energy_ts, color="darkorange", lw=1.5, label="Ising energy")
    _shade(axes[3])
    axes[3].set_ylabel("Energy")
    axes[3].legend(fontsize=16, loc="lower right", frameon=True, framealpha=1.0)

    # ── Panel 4: J coupling + H field ────────────────────────────────────
    j_color = "steelblue"
    h_color = "firebrick"
    axes[4].plot(j_ts, color=j_color, lw=1.5, label="J coupling")
    _shade(axes[4])
    if has_j_peak and j_peak_idx is not None:
        axes[4].axvline(j_peak_idx, color="darkorange", linestyle=":", lw=2.0, alpha=0.9)
        axes[4].plot(j_peak_idx, j_ts[int(j_peak_idx)], "o", color="darkorange",
                     markersize=9, zorder=5, label=f"J peak (idx={j_peak_idx})")
    axes[4].set_ylabel("J Coupling", color=j_color)
    axes[4].tick_params(axis="y", labelcolor=j_color)
    axes[4].set_xlabel("Time bin")

    # Extend y-axis top 30% for legend room
    j_vals = j_ts[np.isfinite(j_ts)]
    if len(j_vals):
        j_lo, j_hi = j_vals.min(), j_vals.max()
        j_rng = j_hi - j_lo if j_hi != j_lo else 1.0
        axes[4].set_ylim(j_lo - 0.05 * j_rng, j_hi + 0.30 * j_rng)

    if h_ts is not None and len(h_ts) > 0:
        ax_h = axes[4].twinx()
        ax_h.plot(h_ts, color=h_color, lw=1.2, alpha=0.75, label="H field")
        h_vals = h_ts[np.isfinite(h_ts)]
        if len(h_vals):
            h_lo, h_hi = h_vals.min(), h_vals.max()
            h_rng = h_hi - h_lo if h_hi != h_lo else 1.0
            ax_h.set_ylim(h_lo - 0.05 * h_rng, h_hi + 0.30 * h_rng)
        if h_peak_idx is not None and not (isinstance(h_peak_idx, float) and np.isnan(h_peak_idx)):
            hidx = int(h_peak_idx)
            ax_h.axvline(hidx, color="salmon", linestyle=":", lw=1.8, alpha=0.9)
            ax_h.plot(hidx, h_ts[hidx], "D", color="salmon", markersize=9,
                      zorder=5, label=f"H peak (idx={hidx})")
        ax_h.set_ylabel("H Field", color=h_color)
        ax_h.tick_params(axis="y", labelcolor=h_color)
        # Merge legends on ax_h (topmost layer); exclude raw time-series labels
        _exclude = {"J coupling", "H field"}
        lines_j, labels_j = axes[4].get_legend_handles_labels()
        lines_h, labels_h = ax_h.get_legend_handles_labels()
        all_lines  = lines_j  + lines_h
        all_labels = labels_j + labels_h
        filtered = [(h, l) for h, l in zip(all_lines, all_labels)
                    if l not in _exclude]
        if filtered:
            fh, fl = zip(*filtered)
            ax_h.legend(fh, fl, fontsize=16, loc="upper right",
                        frameon=True, framealpha=1.0)
    else:
        # No H: show J panel legend with only markers
        _exclude = {"J coupling"}
        lines_j, labels_j = axes[4].get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(lines_j, labels_j)
                    if l not in _exclude]
        if filtered:
            fh, fl = zip(*filtered)
            axes[4].legend(fh, fl, fontsize=16, loc="upper right",
                           frameon=True, framealpha=1.0)

    # ── Panel 5: P(K) ────────────────────────────────────────────────────
    ax_pk = axes[5]
    ax_pk.set_xlabel("K  (active neurons)")
    ax_pk.set_ylabel("P(K)")
    if pk_df is not None and len(pk_df) > 0:
        k_vals = pk_df["K"].values
        ax_pk.plot(k_vals, pk_df["P_data"].values,        "o-", color="black",
                   lw=1.8, markersize=5, label="P_data")
        ax_pk.plot(k_vals, pk_df["P_ising"].values,       "s-", color="crimson",
                   lw=1.8, markersize=5,
                   label=f"P_ising (r={_na(pk_mets['r_ising'])})")
        ax_pk.plot(k_vals, pk_df["P_independent"].values, "^-", color="royalblue",
                   lw=1.8, markersize=5,
                   label=f"P_indep (r={_na(pk_mets['r_independent'])})")
        ax_pk.fill_between(k_vals, pk_df["P_ising"].values,
                           pk_df["P_independent"].values,
                           alpha=0.18, color="mediumpurple",
                           label=f"Ising\u2013indep gap (dist={_na(pk_mets['ising_indep_dist'])})")
        ax_pk.legend(fontsize=16, loc="upper right", frameon=True, framealpha=1.0)

        # Hypothesis annotation (top-left)
        c1 = pk_mets["ising_closer"]
        c2 = pk_mets["ising_indep_dist"] > 0.1
        both = c1 and c2
        tick = "\u2713"; cross = "\u2717"
        annot = (
            f"Hypothesis conditions:\n"
            f"  {tick if c1 else cross} Ising closer to data than independent:\n"
            f"    (r_ising={_na(pk_mets['r_ising'])} \u2265 "
            f"r_indep={_na(pk_mets['r_independent'])} \u2212 0.02)\n"
            f"  {tick if c2 else cross} Ising\u2013indep dist \u2265 0.10\n"
            f"    (dist = {_na(pk_mets['ising_indep_dist'])})\n"
            f"  \u2192 Ising model closer: {tick if both else cross} "
            f"{'MET' if both else 'NOT MET'}"
        )
        ax_pk.annotate(
            annot,
            xy=(0.02, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=16,
            color="darkgreen" if both else "firebrick",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="darkgreen" if both else "firebrick",
                      alpha=0.9),
        )
    else:
        ax_pk.text(0.5, 0.5, "P(K) data unavailable",
                   ha="center", va="center", transform=ax_pk.transAxes,
                   fontsize=16, color="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _savefig(fig, save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 1 — hypothesis scatter
# ---------------------------------------------------------------------------

def _plot_hypothesis_scatter(rows: list, output_dir: str):
    peaks    = [r for r in rows if r["has_j_peak"]]
    no_peaks = [r for r in rows if not r["has_j_peak"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "P(K) hypothesis: Ising\u2013independent divergence vs J peak\n"
        "Hypothesis: peaks appear in top-right (Ising better + differs from indep.)",
        fontweight="bold"
    )

    # Hypothesis region: x > 0.1 AND y > 0
    xlim_hi = max(
        max((r["ising_indep_dist"] for r in rows), default=0.2) * 1.1, 0.5)
    ylim = (
        min((r["r_ising"] - r["r_independent"] for r in rows
             if not np.isnan(r["r_ising"])), default=-0.1) - 0.05,
        max((r["r_ising"] - r["r_independent"] for r in rows
             if not np.isnan(r["r_ising"])), default=0.2) + 0.1,
    )
    # Green shaded hypothesis region (x >= 0.1, y >= 0)
    ax.axhspan(0, max(ylim[1], 0.1), xmin=0.1 / xlim_hi, xmax=1.0,
               color="lightgreen", alpha=0.25, zorder=0, label="Hypothesis region")

    def _scatter(subset, color, label, marker):
        if not subset:
            return
        x = [r["ising_indep_dist"] for r in subset]
        y = [r["r_ising"] - r["r_independent"] for r in subset
             if not np.isnan(r["r_ising"])]
        x = [r["ising_indep_dist"] for r in subset if not np.isnan(r["r_ising"])]
        ax.scatter(x, y, c=color, marker=marker, s=90, alpha=0.85,
                   edgecolors="black", linewidths=0.5, label=label, zorder=3)

    _scatter(peaks,    "crimson",   "J peak",    "o")
    _scatter(no_peaks, "steelblue", "No J peak", "^")

    ax.axvline(0.1, color="gray", linestyle="--", lw=1.2, alpha=0.7,
               label="dist threshold (0.1)")
    ax.axhline(0,   color="gray", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("ising_indep_dist  (sum |P_ising \u2212 P_independent|)")
    ax.set_ylabel("r_ising \u2212 r_independent  (Ising fit advantage)")
    ax.set_xlim(0, xlim_hi)
    ax.legend(fontsize=16, loc="upper left", frameon=True, framealpha=1.0)
    plt.tight_layout()
    path = os.path.join(output_dir, "pk_hypothesis_scatter.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 2 — fit quality bar chart
# ---------------------------------------------------------------------------

def _plot_fit_bars(rows: list, output_dir: str):
    peak_rows   = [r for r in rows if r["has_j_peak"]]
    nopeak_rows = [r for r in rows if not r["has_j_peak"]]

    def _ms(vals):
        a = np.array([v for v in vals if not np.isnan(v)])
        if len(a) == 0:
            return np.nan, np.nan
        return np.mean(a), np.std(a) / np.sqrt(len(a))

    r_ip_m, r_ip_se = _ms([r["r_ising"]       for r in peak_rows])
    r_dp_m, r_dp_se = _ms([r["r_independent"] for r in peak_rows])
    r_in_m, r_in_se = _ms([r["r_ising"]       for r in nopeak_rows])
    r_dn_m, r_dn_se = _ms([r["r_independent"] for r in nopeak_rows])

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("P(K) fit quality: Ising vs Independent by J peak status",
                 fontweight="bold")

    x = np.array([0.0, 0.5, 1.4, 1.9])
    means  = [r_ip_m, r_dp_m, r_in_m, r_dn_m]
    ses    = [r_ip_se, r_dp_se, r_in_se, r_dn_se]
    colors = ["crimson", "salmon", "steelblue", "lightcyan"]
    labels = [
        f"r_ising (peak)",
        f"r_indep (peak)",
        f"r_ising (no peak)",
        f"r_indep (no peak)",
    ]
    for xi, m, se, c, lbl in zip(x, means, ses, colors, labels):
        ax.bar(xi, m, 0.4, yerr=se, capsize=5, color=c,
               edgecolor="black", lw=0.8, label=lbl, alpha=0.90)

    ax.set_xticks([0.25, 1.65])
    ax.set_xticklabels([f"J peak sessions\n(n={len(peak_rows)})",
                        f"No-peak sessions\n(n={len(nopeak_rows)})"])
    ax.set_ylabel("Mean Pearson r  (\u00b1 SE)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, loc="lower right", frameon=True, framealpha=1.0)
    plt.tight_layout()
    path = os.path.join(output_dir, "pk_fit_bars.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 3 — global Mann-Whitney boxplot
# ---------------------------------------------------------------------------

def _plot_global_boxplot(rows: list, output_dir: str):
    """
    Compare r_ising vs r_independent across ALL sessions using an independent
    Mann-Whitney U test (treating the two paired values as independent samples
    when pooled across sessions).
    """
    r_ising = np.array([r["r_ising"]       for r in rows
                        if not np.isnan(r["r_ising"])])
    r_indep = np.array([r["r_independent"] for r in rows
                        if not np.isnan(r["r_independent"])])

    # Mann-Whitney U (independent)
    stat_u, p_mw = mannwhitneyu(r_ising, r_indep, alternative="two-sided")
    sw_ri, sw_ri_p = shapiro(r_ising[:min(len(r_ising), 50)])
    sw_rd, sw_rd_p = shapiro(r_indep[:min(len(r_indep), 50)])

    fig, ax = plt.subplots(figsize=(7, 8))
    fig.suptitle(
        "Global fit quality: Ising vs Independent\n"
        "Independent Wilcoxon\n(Mann-Whitney U)",
        fontweight="bold"
    )

    data   = [r_ising, r_indep]
    colors = ["crimson", "steelblue"]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(data, colors), 1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color="gray", alpha=0.55, s=22, zorder=3)

    # Y-axis limits — enough headroom for bracket + text
    all_vals = np.concatenate([r_ising, r_indep])
    y_min = all_vals.min(); y_max = all_vals.max()
    y_rng = max(y_max - y_min, 0.05)
    y_lo  = y_min - 0.05 * y_rng
    y_hi  = y_max + 0.28 * y_rng
    ax.set_ylim(y_lo, y_hi)

    # Significance bracket (data coords)
    br_y  = y_lo + 0.87 * (y_hi - y_lo)
    br_dh = 0.012 * y_rng
    ax.plot([1, 1, 2, 2], [br_y, br_y + br_dh, br_y + br_dh, br_y],
            color="black", lw=1.3)
    s = _stars(p_mw)
    ax.text(1.5, br_y + 2 * br_dh, s,
            ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.text(1.5, br_y + 5 * br_dh,
            f"p = {p_mw:.4f}  (U={stat_u:.0f})",
            ha="center", va="bottom", fontsize=14)

    # Shapiro-Wilk annotation
    sw_text = (
        f"SW(r_ising): W={sw_ri:.3f} p={sw_ri_p:.4f} "
        f"[{'non-normal \u2713' if sw_ri_p < 0.05 else 'normal'}]\n"
        f"SW(r_indep): W={sw_rd:.3f} p={sw_rd_p:.4f} "
        f"[{'non-normal \u2713' if sw_rd_p < 0.05 else 'normal'}]"
    )
    ax.annotate(sw_text, xy=(0.02, 0.02), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="goldenrod", alpha=0.9))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["r_ising\n(Ising vs data)", "r_independent\n(Indep. vs data)"])
    ax.set_ylabel("Pearson r  (P(K) model vs data)")
    plt.tight_layout()
    path = os.path.join(output_dir, "pk_pearson_boxplot_indep_wilcoxon.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 4 — Ising vs indep split by J-peak (paired Wilcoxon)
# ---------------------------------------------------------------------------

def _plot_ising_vs_indep_by_peak(rows: list, output_dir: str):
    """
    Within J-peak sessions and within no-peak sessions, compare r_ising vs
    r_independent using a paired Wilcoxon signed-rank test (paired because
    both values come from the same session/stim).
    """
    peak_rows   = [(r["r_ising"], r["r_independent"]) for r in rows
                   if r["has_j_peak"]
                   and not np.isnan(r["r_ising"])
                   and not np.isnan(r["r_independent"])]
    nopeak_rows = [(r["r_ising"], r["r_independent"]) for r in rows
                   if not r["has_j_peak"]
                   and not np.isnan(r["r_ising"])
                   and not np.isnan(r["r_independent"])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=False)
    fig.suptitle(
        "Ising vs Independent fit quality \u2014 split by J-peak status\n"
        "Hypothesis: Ising closer when peak present; Independent closer when absent",
        fontweight="bold", y=1.02
    )

    groups = [
        (axes[0], peak_rows,   f"J-peak sessions  (n={len(peak_rows)})"),
        (axes[1], nopeak_rows, f"No-peak sessions  (n={len(nopeak_rows)})"),
    ]

    for ax, pairs, title in groups:
        ax.set_title(title, fontweight="bold", pad=14)

        if not pairs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue

        ri = np.array([p[0] for p in pairs])
        rd = np.array([p[1] for p in pairs])
        diffs = ri - rd

        # Paired Wilcoxon signed-rank test
        try:
            wstat, p_w = wilcoxon(ri, rd)
        except Exception:
            wstat, p_w = np.nan, np.nan

        # Shapiro-Wilk on differences
        try:
            sw_w, sw_p = shapiro(diffs[:50])
        except Exception:
            sw_w, sw_p = np.nan, np.nan

        bp = ax.boxplot([ri, rd], patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        widths=0.5)
        box_color = "salmon"
        for patch in bp["boxes"]:
            patch.set_facecolor(box_color); patch.set_alpha(0.65)

        rng = np.random.default_rng(7)
        for i, vals in enumerate([ri, rd], 1):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="salmon", alpha=0.5, s=22, zorder=3,
                       edgecolors="black", linewidths=0.4)

        # Set y-axis limits with enough headroom for bracket + stats text
        all_vals = np.concatenate([ri, rd])
        y_min = all_vals.min()
        y_max = all_vals.max()
        y_rng = max(y_max - y_min, 0.05)
        y_lo  = y_min - 0.08 * y_rng
        # Top headroom: 35% of range for bracket + text + Shapiro at bottom
        y_hi  = y_max + 0.35 * y_rng
        ax.set_ylim(y_lo, y_hi)

        # Bracket connecting the two boxes (in data coords)
        # Place bracket at 88% of the y-axis visible range
        br_y  = y_lo + 0.88 * (y_hi - y_lo)
        br_dh = 0.015 * y_rng          # vertical tick height
        ax.plot([1, 1, 2, 2],
                [br_y, br_y + br_dh, br_y + br_dh, br_y],
                color="black", lw=1.3, clip_on=False)

        # Significance stars just above bracket
        s = _stars(p_w) if not np.isnan(p_w) else "n/a"
        ax.text(1.5, br_y + 2 * br_dh, s,
                ha="center", va="bottom", fontsize=18, fontweight="bold")

        # Wilcoxon stats text inside plot — upper portion, centred
        med_diff = float(np.median(diffs))
        stats_line = f"Wilcoxon W={wstat:.0f}  p={p_w:.4f}\nmedian diff={med_diff:+.3f}"
        ax.text(0.5, 0.80, stats_line,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.0))

        # Shapiro-Wilk annotation in a coloured box at the bottom
        sw_text = (
            f"Shapiro-Wilk (diff): W={sw_w:.3f} p={sw_p:.4f} "
            f"[{'non-normal \u2713' if sw_p < 0.05 else 'normal'}]"
        )
        ax.annotate(sw_text, xy=(0.02, 0.02), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="goldenrod", alpha=0.9))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["r_ising\n(Ising vs data)",
                             "r_independent\n(Indep. vs data)"])
        ax.set_ylabel("Pearson r  (P(K) model vs data)")

    fig.tight_layout()
    path = os.path.join(output_dir, "pk_ising_vs_indep_by_peak.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 5 — stim comparison scatter
# ---------------------------------------------------------------------------

def _plot_stim_comparison(rows: list, output_dir: str):
    stims = sorted(set(r["stim"] for r in rows))
    if len(stims) < 2:
        return

    s0, s1 = stims[0], stims[1]
    by_sid_s0 = {r["session_id"]: r for r in rows if r["stim"] == s0}
    by_sid_s1 = {r["session_id"]: r for r in rows if r["stim"] == s1}
    common = sorted(set(by_sid_s0) & set(by_sid_s1))
    if not common:
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle(
        "Baseline vs Perturbation: collective state\n"
        "(positive = Ising wins, negative = independent wins)",
        fontweight="bold"
    )

    xs, ys = [], []
    for sid in common:
        r0 = by_sid_s0[sid]; r1 = by_sid_s1[sid]
        x = r0["collective_score"]; y = r1["collective_score"]
        xs.append(x); ys.append(y)
        p0 = r0["has_j_peak"]; p1 = r1["has_j_peak"]
        if p0 and p1:
            c = "limegreen"; lbl = "J peak in both stims"
        elif not p0 and not p1:
            c = "gray";      lbl = "No peak in either stim"
        else:
            c = "darkorange"; lbl = "Peak in one stim only"
        ax.scatter(x, y, color=c, s=70, alpha=0.85, edgecolors="black",
                   linewidths=0.5, zorder=3, label=lbl)
        ax.annotate(sid, (x, y), fontsize=12, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points",
                    color="dimgray", alpha=0.8)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=15, loc="lower right",
              frameon=True, framealpha=1.0)

    all_vals = xs + ys
    lim = max(abs(min(all_vals, default=0)), abs(max(all_vals, default=1))) * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8,
            linestyle=":", alpha=0.6)
    ax.set_xlabel(f"Collective score  \u2014  Stim {s0} (baseline)")
    ax.set_ylabel(f"Collective score  \u2014  Stim {s1} (perturbation)")
    plt.tight_layout()
    path = os.path.join(output_dir, "pk_stim_comparison.png")
    _savefig(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _write_text_report(rows: list, output_dir: str, args):
    n_peak       = sum(1 for r in rows if r["has_j_peak"])
    n_collective = sum(1 for r in rows if r.get("is_collective", False))
    n_both       = sum(1 for r in rows if r["has_j_peak"]
                       and r.get("is_collective", False))

    lines = [
        "PK-J ANALYSIS REPORT",
        "=" * 60,
        f"Data folder : {getattr(args, 'data_folder', 'n/a')}",
        f"Window      : [{args.window[0]}, {args.window[1]}]",
        f"Threshold   : {args.peak_threshold}",
        f"Smooth sigma: {args.smooth_sigma}",
        "",
        f"Total sessions : {len(rows)}",
        f"J peak         : {n_peak}  ({100*n_peak/max(len(rows),1):.1f}%)",
        f"Collective     : {n_collective}  ({100*n_collective/max(len(rows),1):.1f}%)",
        f"Both           : {n_both}",
        "",
        "TIMING BREAKDOWN",
        "-" * 40,
    ]
    for timing in ("leads", "lags", "simultaneous", "no_peak"):
        n = sum(1 for r in rows if r.get("timing") == timing)
        lines.append(f"  {timing:<14}: {n}")

    path = os.path.join(output_dir, "pk_summary.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Text report \u2192 {path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(rows: list, output_dir: str):
    col_order = [
        "session_id", "stim",
        "has_j_peak", "j_peak_idx", "j_peak_ratio", "timing",
        "has_h_peak", "h_peak_idx",
        "accel_peak_idx", "vel_peak_idx",
        "fr_peak_idx", "fr_trough_idx",
        "is_collective",
        "r_ising", "r_independent", "r_ising_vs_indep",
        "ising_indep_dist", "ising_closer", "collective_score",
    ]
    df = pd.DataFrame(rows)
    extra = [c for c in df.columns if c not in col_order]
    df = df[[c for c in col_order if c in df.columns] + extra]
    df = df.sort_values(["session_id", "stim"]).reset_index(drop=True)
    path = os.path.join(output_dir, "pk_j_summary.csv")
    df.to_csv(path, index=False)
    print(f"  CSV summary \u2192 {path}")


# ---------------------------------------------------------------------------
# Per-session processing task (parallelised)
# ---------------------------------------------------------------------------

def _process_task(args_tuple):
    (session_path, session_id, stim, w_lo, w_hi,
     peak_threshold, smooth_sigma, output_dir) = args_tuple

    try:
        df = pd.read_csv(session_path)
        stim_df = df[df["stim"] == stim] if "stim" in df.columns else df
        if stim_df.empty:
            return None

        vel_ts    = _velocity_ts(stim_df)
        accel_ts  = _acceleration_ts(stim_df)
        j_ts      = _mean_ts(stim_df, "j")
        h_ts      = _mean_ts(stim_df, "h")   if "h"           in stim_df.columns else np.array([])
        fr_ts     = _mean_ts(stim_df, "firing_rate") if "firing_rate" in stim_df.columns else np.array([])
        energy_ts = _mean_ts(stim_df, "energy")      if "energy"       in stim_df.columns else np.array([])

        if len(vel_ts) == 0 or len(j_ts) == 0:
            return None

        n  = len(vel_ts)
        lo = max(0, w_lo)
        hi = min(n, w_hi)

        # Kinematic peaks
        vel_win   = vel_ts[lo:hi]
        accel_win = accel_ts[lo:hi] if len(accel_ts) >= hi else np.array([])
        if len(vel_win) == 0:
            return None
        vel_peak_idx   = lo + int(np.argmax(np.abs(vel_win)))
        accel_peak_idx = (lo + int(np.argmax(np.abs(accel_win)))
                          if len(accel_win) > 0 else vel_peak_idx)

        # J peak
        has_j_peak, j_peak_idx, _, j_peak_ratio = _j_peak_detect(
            j_ts, lo, hi, threshold_ratio=peak_threshold, smooth_sigma=smooth_sigma)

        # H peak
        if len(h_ts) > 0:
            has_h_peak, h_peak_idx, _, _ = _signal_extremum_detect(
                h_ts, lo, hi, smooth_sigma=smooth_sigma)
        else:
            has_h_peak, h_peak_idx = False, None

        # FR peak and trough
        if len(fr_ts) > 0:
            has_fr_peak,   fr_peak_idx,   _, _ = _signal_extremum_detect(
                fr_ts, lo, hi, kind="peak",   smooth_sigma=smooth_sigma)
            has_fr_trough, fr_trough_idx, _, _ = _signal_extremum_detect(
                fr_ts, lo, hi, kind="trough", smooth_sigma=smooth_sigma)
        else:
            has_fr_peak,   fr_peak_idx   = False, None
            has_fr_trough, fr_trough_idx = False, None

        # Timing classification
        if has_j_peak and j_peak_idx is not None:
            lag_a = j_peak_idx - accel_peak_idx
            lag_v = j_peak_idx - vel_peak_idx
            if abs(lag_a) <= 2 or abs(lag_v) <= 2:
                timing = "simultaneous"
            elif j_peak_idx < min(accel_peak_idx, vel_peak_idx):
                timing = "leads"
            else:
                timing = "lags"
        else:
            timing = "no_peak"

        # P(K) metrics
        stim_dir = os.path.dirname(session_path)
        pk_df    = _load_pk(stim_dir)
        if pk_df is not None:
            pk_mets = _pk_metrics(pk_df)
        else:
            pk_mets = {
                "r_ising": np.nan, "r_independent": np.nan,
                "r_ising_vs_indep": np.nan, "ising_indep_dist": np.nan,
                "ising_closer": False, "collective_score": np.nan,
            }

        is_collective = bool(pk_mets["ising_closer"]
                             and pk_mets["ising_indep_dist"] > 0.1)

        # Save per-session plot
        subdir = os.path.join(output_dir, timing)
        fname  = f"stim{stim}_{session_id}.png"
        _plot_session(
            session_id=session_id, stim=stim,
            vel_ts=vel_ts, accel_ts=accel_ts, fr_ts=fr_ts,
            energy_ts=energy_ts, j_ts=j_ts,
            pk_df=pk_df, pk_mets=pk_mets,
            w_lo=lo, w_hi=hi,
            vel_peak_idx=vel_peak_idx, accel_peak_idx=accel_peak_idx,
            has_j_peak=has_j_peak,
            j_peak_idx=j_peak_idx if has_j_peak else None,
            j_peak_ratio=float(j_peak_ratio),
            timing=timing,
            save_path=os.path.join(subdir, fname),
            smooth_sigma=smooth_sigma,
            h_ts=h_ts if len(h_ts) > 0 else None,
            h_peak_idx=h_peak_idx    if has_h_peak    else None,
            fr_peak_idx=fr_peak_idx   if has_fr_peak   else None,
            fr_trough_idx=fr_trough_idx if has_fr_trough else None,
        )

        row = {
            "session_id":     session_id,
            "stim":           stim,
            "has_j_peak":     has_j_peak,
            "j_peak_idx":     int(j_peak_idx)   if has_j_peak  and j_peak_idx   is not None else np.nan,
            "j_peak_ratio":   float(j_peak_ratio),
            "timing":         timing,
            "has_h_peak":     has_h_peak,
            "h_peak_idx":     int(h_peak_idx)   if has_h_peak  and h_peak_idx   is not None else np.nan,
            "accel_peak_idx": int(accel_peak_idx),
            "vel_peak_idx":   int(vel_peak_idx),
            "fr_peak_idx":    int(fr_peak_idx)   if has_fr_peak   and fr_peak_idx   is not None else np.nan,
            "fr_trough_idx":  int(fr_trough_idx) if has_fr_trough and fr_trough_idx is not None else np.nan,
            "is_collective":  is_collective,
            **pk_mets,
        }
        print(f"  \u2713 {session_id}  stim={stim}  "
              f"peak={'Y' if has_j_peak else 'n'}  timing={timing:<13}  "
              f"r_ising={pk_mets['r_ising']:.3f}  "
              f"dist={pk_mets['ising_indep_dist']:.3f}")
        return row

    except Exception as e:
        print(f"  [WARN] {session_id} stim={stim}: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="P(K) \u00d7 J-peak hypothesis analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_folder", required=True,
                   help="Root directory containing per_reach_state.csv files.")
    p.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"),
                   default=[350, 475])
    p.add_argument("--stim_min", type=int, default=0)
    p.add_argument("--stim_max_exclusive", type=int, default=3)
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Restrict to these session IDs (6-digit strings).")
    p.add_argument("--peak_threshold", type=float, default=1.75)
    p.add_argument("--smooth_sigma",   type=float, default=5.0)
    p.add_argument("--output_dir", default="./notes/pk_analysis")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    if not os.path.isdir(data_folder):
        print(f"Error: {data_folder} is not a directory", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    w_lo, w_hi = args.window
    stim_range = range(args.stim_min, args.stim_max_exclusive)
    whitelist  = set(args.sessions) if args.sessions else None

    print(f"Scanning {data_folder} \u2026")
    all_paths = find_file_recursive(data_folder, "per_reach_state.csv", verbose=True)
    if not all_paths:
        print("No per_reach_state.csv found.", file=sys.stderr)
        return 1

    tasks = []
    for path in all_paths:
        stim = _stim_from_path(path)
        if stim is None or stim not in stim_range:
            continue
        sid = _session_id_from_path(path)
        if whitelist and sid not in whitelist:
            continue
        tasks.append((path, sid, stim, w_lo, w_hi,
                      args.peak_threshold, args.smooth_sigma, args.output_dir))

    if not tasks:
        print("No tasks generated \u2014 check --data_folder and stim range.")
        return 1

    print(f"Processing {len(tasks)} session/stim pairs \u2026\n")
    pool_kw = {} if args.workers is None else {"processes": args.workers}
    rows = []
    with Pool(**pool_kw) as pool:
        for row in pool.imap_unordered(_process_task, tasks):
            if row is not None:
                rows.append(row)

    if not rows:
        print("No rows produced.", file=sys.stderr)
        return 1

    print(f"\n{len(rows)} rows processed. Generating summary figures \u2026")
    _plot_hypothesis_scatter(rows, args.output_dir)
    _plot_fit_bars(rows, args.output_dir)
    _plot_global_boxplot(rows, args.output_dir)
    _plot_ising_vs_indep_by_peak(rows, args.output_dir)
    _plot_stim_comparison(rows, args.output_dir)
    _write_text_report(rows, args.output_dir, args)
    _write_csv(rows, args.output_dir)

    print(f"\nDone. Results in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
