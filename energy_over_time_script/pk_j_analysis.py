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

# Each row of per_reach_state.csv is one 10 ms time bin; x/y/z positions are
# recorded in cm. Used to convert bin indices to ms and position deltas to
# cm/s, cm/s^2 for axis labels and unit-correct kinematic traces.
BIN_MS = 10.0

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
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: str, **kwargs):
    """Save *fig* as PNG and SVG (svg.fonttype='none' for Affinity Designer)."""
    fig.savefig(path, **kwargs)
    svg_path = os.path.splitext(path)[0] + ".svg"
    svg_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(svg_path, **svg_kwargs)


def _tighten(fig, w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04):
    """Compact spacing shared by every figure in this script.

    Uses the constrained-layout engine rather than ``tight_layout`` because the
    latter ignores ``suptitle``, which pushed multi-line titles down onto the
    y-axis label.
    """
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)


def _bottom_note(fig, text, fontsize=15):
    """Attach a supplementary note under the axes.

    Uses ``supxlabel`` so the constrained-layout engine reserves room for it;
    placing these boxes inside the axes corner made them collide with the tick
    labels.
    """
    t = fig.supxlabel(text, fontsize=fontsize)
    t.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                    edgecolor="goldenrod", alpha=0.9))
    return t


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


def _velocity_ts(df: pd.DataFrame, bin_ms: float = BIN_MS) -> np.ndarray:
    """X velocity in cm/s (x is recorded in cm; bins are ``bin_ms`` wide)."""
    dt_s = bin_ms / 1000.0
    parts = []
    for _, grp in df.groupby("reach_idx"):
        x = grp["x"].values
        parts.append(np.diff(x, prepend=x[0]) / dt_s)
    if not parts:
        return np.array([])
    n = min(len(p) for p in parts)
    return np.mean([p[:n] for p in parts], axis=0)


def _acceleration_ts(df: pd.DataFrame, bin_ms: float = BIN_MS) -> np.ndarray:
    """X acceleration in cm/s^2."""
    dt_s = bin_ms / 1000.0
    v = _velocity_ts(df, bin_ms)
    return np.diff(v, prepend=v[0]) / dt_s if len(v) else np.array([])


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
# Per-session 5-panel plot
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
    5-panel figure per session (no legends -- markers/colors are explained in
    the figure caption):
      0: X velocity (cm/s)
      1: X acceleration (cm/s^2)
      2: Firing rate + mean line + peak/trough markers
      3: Ising energy
      4: J coupling + H field (twin y-axis) + peak markers
    Time axis is in ms (BIN_MS per bin); the P(K) panel previously shown here
    lives in the summary figures instead.
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

    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=False,
                             layout="constrained")
    _tighten(fig, hspace=0.03)
    fig.suptitle(title, fontsize=21, fontweight="bold")

    def _t(idx_array):
        return np.asarray(idx_array) * BIN_MS

    w_lo_ms, w_hi_ms = w_lo * BIN_MS, w_hi * BIN_MS
    accel_peak_ms = accel_peak_idx * BIN_MS
    vel_peak_ms   = vel_peak_idx * BIN_MS

    def _shade(ax):
        ax.axvspan(w_lo_ms, w_hi_ms, color="gold", alpha=0.12, zorder=0)
        ax.axvline(w_lo_ms, color="goldenrod", linestyle="--", lw=0.9, alpha=0.5)
        ax.axvline(w_hi_ms, color="goldenrod", linestyle="--", lw=0.9, alpha=0.5)
        ax.axvline(accel_peak_ms, color="green", linestyle="--", lw=1.4, alpha=0.6)
        ax.axvline(vel_peak_ms,   color="deepskyblue", linestyle="--", lw=1.4, alpha=0.6)

    def _panel_letter(ax, letter):
        ax.text(0.01, 0.97, letter, transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="top", ha="left", zorder=6)

    panel_letters = "ABCDE"

    # ── Panel 0: Velocity ────────────────────────────────────────────────
    axes[0].plot(_t(np.arange(len(vel_ts))), vel_ts, color="navy", lw=1.5)
    _shade(axes[0])
    axes[0].set_ylabel("X Velocity (cm/s)")
    _panel_letter(axes[0], panel_letters[0])

    # ── Panel 1: Acceleration ────────────────────────────────────────────
    axes[1].plot(_t(np.arange(len(accel_ts))), accel_ts, color="steelblue", lw=1.5)
    _shade(axes[1])
    axes[1].set_ylabel(r"X Acceleration (cm/s$^2$)")
    _panel_letter(axes[1], panel_letters[1])

    # ── Panel 2: Firing rate ─────────────────────────────────────────────
    axes[2].plot(_t(np.arange(len(fr_ts))), fr_ts, color="darkgreen", lw=1.5)
    fr_mean = np.nanmean(fr_ts) if len(fr_ts) > 0 else np.nan
    axes[2].axhline(fr_mean, color="darkgreen", linestyle="--", lw=1.2, alpha=0.5)
    _shade(axes[2])
    if fr_peak_idx is not None and not (isinstance(fr_peak_idx, float) and np.isnan(fr_peak_idx)):
        idx = int(fr_peak_idx)
        axes[2].axvline(_t(idx), color="limegreen", linestyle=":", lw=1.8, alpha=0.9)
        axes[2].plot(_t(idx), fr_ts[idx], "^", color="limegreen", markersize=10, zorder=5)
    if fr_trough_idx is not None and not (isinstance(fr_trough_idx, float) and np.isnan(fr_trough_idx)):
        idx = int(fr_trough_idx)
        axes[2].axvline(_t(idx), color="darkred", linestyle=":", lw=1.8, alpha=0.9)
        axes[2].plot(_t(idx), fr_ts[idx], "v", color="darkred", markersize=10, zorder=5)
    axes[2].set_ylabel("Firing Rate")
    _panel_letter(axes[2], panel_letters[2])

    # ── Panel 3: Energy ──────────────────────────────────────────────────
    axes[3].plot(_t(np.arange(len(energy_ts))), energy_ts, color="darkorange", lw=1.5)
    _shade(axes[3])
    axes[3].set_ylabel("Energy")
    _panel_letter(axes[3], panel_letters[3])

    # ── Panel 4: J coupling + H field ────────────────────────────────────
    j_color = "steelblue"
    h_color = "firebrick"
    axes[4].plot(_t(np.arange(len(j_ts))), j_ts, color=j_color, lw=1.5)
    _shade(axes[4])
    _panel_letter(axes[4], panel_letters[4])
    if has_j_peak and j_peak_idx is not None:
        axes[4].axvline(_t(j_peak_idx), color="darkorange", linestyle=":", lw=2.0, alpha=0.9)
        axes[4].plot(_t(j_peak_idx), j_ts[int(j_peak_idx)], "o", color="darkorange",
                     markersize=9, zorder=5)
    axes[4].set_ylabel("J Coupling", color=j_color)
    axes[4].tick_params(axis="y", labelcolor=j_color)
    axes[4].set_xlabel("Time (ms)")

    j_vals = j_ts[np.isfinite(j_ts)]
    if len(j_vals):
        j_lo, j_hi = j_vals.min(), j_vals.max()
        j_rng = j_hi - j_lo if j_hi != j_lo else 1.0
        axes[4].set_ylim(j_lo - 0.05 * j_rng, j_hi + 0.10 * j_rng)

    if h_ts is not None and len(h_ts) > 0:
        ax_h = axes[4].twinx()
        ax_h.plot(_t(np.arange(len(h_ts))), h_ts, color=h_color, lw=1.2, alpha=0.75)
        h_vals = h_ts[np.isfinite(h_ts)]
        if len(h_vals):
            h_lo, h_hi = h_vals.min(), h_vals.max()
            h_rng = h_hi - h_lo if h_hi != h_lo else 1.0
            ax_h.set_ylim(h_lo - 0.05 * h_rng, h_hi + 0.10 * h_rng)
        if h_peak_idx is not None and not (isinstance(h_peak_idx, float) and np.isnan(h_peak_idx)):
            hidx = int(h_peak_idx)
            ax_h.axvline(_t(hidx), color="salmon", linestyle=":", lw=1.8, alpha=0.9)
            ax_h.plot(_t(hidx), h_ts[hidx], "D", color="salmon", markersize=9, zorder=5)
        ax_h.set_ylabel("H Field", color=h_color)
        ax_h.tick_params(axis="y", labelcolor=h_color)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _savefig(fig, save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot 1 — hypothesis scatter
# ---------------------------------------------------------------------------

def _plot_hypothesis_scatter(rows: list, output_dir: str):
    peaks    = [r for r in rows if r["has_j_peak"]]
    no_peaks = [r for r in rows if not r["has_j_peak"]]

    fig, ax = plt.subplots(figsize=(12, 7.5), layout="constrained")
    _tighten(fig)
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
    # Without this the axes autoscaled to the shaded region instead of the
    # points, leaving the top half of the panel empty.
    ax.set_ylim(*ylim)
    ax.legend(fontsize=17, loc="upper left", frameon=True, framealpha=1.0)
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

    fig, ax = plt.subplots(figsize=(10.5, 7), layout="constrained")
    _tighten(fig)
    fig.suptitle("P(K) fit quality: Ising vs Independent by J peak status",
                 fontweight="bold")

    # Groups pulled closer together (was a 0.9 gap against 0.5 within-group)
    x = np.array([0.0, 0.45, 1.15, 1.60])
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

    ax.set_xticks([0.225, 1.375])
    ax.set_xticklabels([f"J peak sessions\n(n={len(peak_rows)})",
                        f"No-peak sessions\n(n={len(nopeak_rows)})"])
    ax.set_ylabel("Mean Pearson r  (\u00b1 SE)")
    # Headroom so the legend clears the bars instead of covering them.
    ax.set_ylim(0, 1.32)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(fontsize=15, loc="upper center", ncol=2, frameon=True,
              framealpha=1.0)
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

    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    _tighten(fig)
    fig.suptitle(
        "Global fit quality: Ising vs Independent\n"
        "Independent Wilcoxon (Mann-Whitney U)",
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
    y_hi  = y_max + 0.40 * y_rng
    ax.set_ylim(y_lo, y_hi)

    # Significance bracket (data coords)
    br_y  = y_lo + 0.84 * (y_hi - y_lo)
    br_dh = 0.012 * y_rng
    ax.plot([1, 1, 2, 2], [br_y, br_y + br_dh, br_y + br_dh, br_y],
            color="black", lw=1.3)
    # Stars above stats, matching _plot_ising_vs_indep_by_peak.
    ax.text(0.5, 0.99, _stars(p_mw), transform=ax.transAxes,
            ha="center", va="top", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.925, f"p = {p_mw:.4f}  (U={stat_u:.0f})",
            transform=ax.transAxes, ha="center", va="top", fontsize=16)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["r_ising\n(Ising vs data)", "r_independent\n(Indep. vs data)"])
    ax.set_ylabel("Pearson r  (P(K) model vs data)")

    _bottom_note(fig, (
        f"SW(r_ising): W={sw_ri:.3f} p={sw_ri_p:.4f} "
        f"[{'non-normal \u2713' if sw_ri_p < 0.05 else 'normal'}]    "
        f"SW(r_indep): W={sw_rd:.3f} p={sw_rd_p:.4f} "
        f"[{'non-normal \u2713' if sw_rd_p < 0.05 else 'normal'}]"
    ))
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

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.5), sharey=False,
                             layout="constrained")
    _tighten(fig, h_pad=0.06, wspace=0.05)
    fig.suptitle(
        "Ising vs Independent fit quality \u2014 split by J-peak status\n"
        "Hypothesis: Ising closer when peak present; Independent closer when absent",
        fontweight="bold"
    )

    sw_notes = []
    groups = [
        (axes[0], peak_rows,   f"J-peak sessions  (n={len(peak_rows)})"),
        (axes[1], nopeak_rows, f"No-peak sessions  (n={len(nopeak_rows)})"),
    ]

    for ax, pairs, title in groups:
        ax.set_title(title, fontweight="bold", pad=10)

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
        # Top headroom: 55% of range for bracket + stats text
        y_hi  = y_max + 0.55 * y_rng
        ax.set_ylim(y_lo, y_hi)

        # Stacked top to bottom in the headroom: stars, stats, bracket.
        br_y  = y_lo + 0.78 * (y_hi - y_lo)
        br_dh = 0.015 * y_rng          # vertical tick height
        ax.plot([1, 1, 2, 2],
                [br_y, br_y + br_dh, br_y + br_dh, br_y],
                color="black", lw=1.3)

        s = _stars(p_w) if not np.isnan(p_w) else "n/a"
        ax.text(0.5, 0.99, s, transform=ax.transAxes,
                ha="center", va="top", fontsize=22, fontweight="bold")

        med_diff = float(np.median(diffs))
        stats_line = f"Wilcoxon W={wstat:.0f}  p={p_w:.4f}\nmedian diff={med_diff:+.3f}"
        ax.text(0.5, 0.91, stats_line,
                transform=ax.transAxes, ha="center", va="top", fontsize=16)

        # Collected into a single caption below both panels; boxed inside the
        # axes these ran off the bottom edge and clipped the tick labels.
        sw_notes.append(
            f"{title.split('  (')[0]}: W={sw_w:.3f} p={sw_p:.4f} "
            f"[{'non-normal \u2713' if sw_p < 0.05 else 'normal'}]"
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["r_ising\n(Ising vs data)",
                             "r_independent\n(Indep. vs data)"])
        # Y-label only on the left panel — the right panel keeps its own
        # ticks (independent scale) but drops the duplicate label text.
        if ax is axes[0]:
            ax.set_ylabel("Pearson r  (P(K) model vs data)")

    if sw_notes:
        _bottom_note(fig, "Shapiro-Wilk (diff)   " + "    ".join(sw_notes),
                     fontsize=14)

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

    fig, ax = plt.subplots(figsize=(10.5, 9), layout="constrained")
    _tighten(fig)
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

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    # Points sit along the diagonal, so the upper-left corner is the only
    # reliably empty region; "lower right" covered the session labels.
    ax.legend(seen.values(), seen.keys(), fontsize=16, loc="upper left",
              frameon=True, framealpha=1.0)

    # Square limits spanning the data and zero (so the sign quadrants stay
    # readable) rather than a symmetric range that left one quadrant empty.
    all_vals = (xs + ys) or [0.0, 1.0]
    lo = min(min(all_vals), 0.0)
    hi = max(max(all_vals), 0.0)
    pad = 0.12 * max(hi - lo, 0.1)
    lo -= pad
    hi += pad
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8,
            linestyle=":", alpha=0.6)
    ax.set_xlabel(f"Collective score  \u2014  Stim {s0} (baseline)")
    ax.set_ylabel(f"Collective score  \u2014  Stim {s1} (perturbation)")

    # Session labels, stacked outward where points crowd together. Most
    # sessions sit in a tight clump near the origin and a single fixed offset
    # printed them on top of one another.
    # Greedy placement against actual label extents: walk top-down and lift
    # each label until its box clears everything already placed.
    fig.canvas.draw()
    fs = 13
    lab_h = fs * 1.35
    to_points = 72.0 / fig.dpi
    boxes = []
    for sid, x, y in sorted(zip(common, xs, ys), key=lambda t: (-t[2], t[1])):
        text = str(sid)
        px, py = ax.transData.transform((x, y))
        px *= to_points
        py *= to_points
        w = 0.60 * fs * len(text)
        dx, dy = 5.0, 3.0
        for _ in range(40):
            bx, by = px + dx, py + dy
            if not any(bx < ox + ow and ox < bx + w
                       and by < oy + oh and oy < by + lab_h
                       for ox, oy, ow, oh in boxes):
                break
            dy += lab_h * 0.9
        boxes.append((px + dx, py + dy, w, lab_h))
        # Lifted labels need a leader line, otherwise it is not clear which
        # point in the cluster they belong to.
        arrow = (dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.55,
                      shrinkA=0, shrinkB=3)
                 if dy > 3.0 + lab_h else None)
        ax.annotate(text, (x, y), fontsize=fs, ha="left", va="bottom",
                    xytext=(dx, dy), textcoords="offset points",
                    color="dimgray", alpha=0.85, annotation_clip=False,
                    arrowprops=arrow)

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
