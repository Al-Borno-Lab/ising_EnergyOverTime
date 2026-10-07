#!/usr/bin/env python
# coding: utf-8
"""
Find J peaks inside the max-velocity window for manually supplied session IDs.

Workflow per session / stimulus:
  1. Recurse through --data_folder to find all per_reach_state.csv files.
  2. Match each file against the session IDs supplied via --sessions
     (last 6 characters before "_results" in the path, same convention as
     arbitration_j_many.py).
  3. Compute mean x-velocity, x-acceleration, and firing rate across reaches.
  4. Locate max-velocity index; define search window ±half_window bins around it.
  5. Apply an optional low-pass (Butterworth) filter to J.
  6. Find ALL local maxima of filtered J inside the velocity window (scipy
     find_peaks).  Mark every peak on the plot; report lag for the HIGHEST one.
  7. Read model_quality_summary_P_K.csv from the same directory to compute:
       r_ising        = Pearson r(P_data, P_ising)
       r_independent  = Pearson r(P_data, P_independent)
  8. Save a 4-panel plot (velocity | acceleration | firing rate | J) and a
     consolidated summary CSV with N_neurons, stim, peak info, and model fit.

Usage
-----
    python peak_near_max_velocity.py \\
        --data_folder /path/to/results_root \\
        --sessions 123456 789ABC DEF012 \\
        --stim_min 0 --stim_max_exclusive 3 \\
        --half_window 60 \\
        --cutoff 0.2 --filter_order 4 \\
        --output_dir ./peak_results

Run  python peak_near_max_velocity.py --help  for all options.
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
from scipy.signal import butter, filtfilt, find_peaks

from src.util import find_file_recursive


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------

def _mean_timeseries(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mean time series of *col* across reaches."""
    ts = []
    for _, group in df.groupby("reach_idx"):
        if col in group.columns:
            ts.append(group[col].values)
    if not ts:
        return np.array([])
    min_len = min(len(a) for a in ts)
    return np.mean([a[:min_len] for a in ts], axis=0)


def _velocity_timeseries(df: pd.DataFrame) -> np.ndarray:
    """Mean x-velocity time series (finite difference of x position)."""
    vels = []
    for _, group in df.groupby("reach_idx"):
        x = group["x"].values
        vels.append(np.diff(x, prepend=x[0]))
    if not vels:
        return np.array([])
    min_len = min(len(v) for v in vels)
    return np.mean([v[:min_len] for v in vels], axis=0)


def _acceleration_timeseries(df: pd.DataFrame) -> np.ndarray:
    """Mean x-acceleration time series (second finite difference of x position)."""
    vel = _velocity_timeseries(df)
    return np.diff(vel, prepend=vel[0]) if len(vel) else np.array([])


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _lowpass_filter(signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    cutoff : normalised frequency in (0, 1) where 1.0 == Nyquist.
             0.08 keeps features wider than ~12 samples.
    """
    if len(signal) < 3 * (order + 1):
        return signal.astype(float)
    b, a = butter(order, cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal.astype(float))


def _gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel smoothing (alternative to Butterworth)."""
    if sigma <= 0 or len(arr) < 3:
        return arr.astype(float)
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr.astype(float), kernel, mode="same")


def _apply_filter(
    signal: np.ndarray,
    cutoff: float,
    filter_order: int,
    use_gaussian: bool,
    smooth_sigma: float,
) -> np.ndarray:
    if use_gaussian:
        return _gaussian_smooth(signal, smooth_sigma)
    if cutoff > 0:
        return _lowpass_filter(signal, cutoff, filter_order)
    return signal.astype(float)


# ---------------------------------------------------------------------------
# Model quality helper
# ---------------------------------------------------------------------------

def _read_model_quality(csv_dir: str) -> dict:
    """
    Read model_quality_summary_P_K.csv and _P_K_metadata.csv from *csv_dir*.

    Returns a dict with:
        n_neurons      : int   (from metadata; nan if unavailable)
        r_ising        : float Pearson r(P_data, P_ising)
        r_independent  : float Pearson r(P_data, P_independent)
    All values are nan if the files are not found or cannot be parsed.
    """
    result = {"n_neurons": np.nan, "r_ising": np.nan, "r_independent": np.nan}

    pk_path   = os.path.join(csv_dir, "model_quality_summary_P_K.csv")
    meta_path = os.path.join(csv_dir, "model_quality_summary_P_K_metadata.csv")

    if os.path.isfile(meta_path):
        try:
            meta = pd.read_csv(meta_path)
            if "N_neurons" in meta.columns:
                result["n_neurons"] = int(meta["N_neurons"].iloc[0])
        except Exception:
            pass

    if os.path.isfile(pk_path):
        try:
            pk = pd.read_csv(pk_path)
            required = {"P_data", "P_ising", "P_independent"}
            if required.issubset(pk.columns) and len(pk) > 1:
                p_data = pk["P_data"].values.astype(float)
                p_ising = pk["P_ising"].values.astype(float)
                p_indep = pk["P_independent"].values.astype(float)
                result["r_ising"]       = float(np.corrcoef(p_data, p_ising)[0, 1])
                result["r_independent"] = float(np.corrcoef(p_data, p_indep)[0, 1])
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Core analysis — find ALL J peaks inside the velocity window
# ---------------------------------------------------------------------------

def find_j_peaks_near_velocity(
    vel_ts: np.ndarray,
    j_ts: np.ndarray,
    j_filt: np.ndarray,
    half_window: int = 60,
    min_prominence: float = 0.0,
) -> dict:
    """
    Find ALL local J maxima inside the ±half_window velocity window, and
    identify the highest one.

    Returns
    -------
    dict with keys:
        vel_peak_idx      : index of max |velocity|
        vel_peak_value    : velocity value there
        win_lo, win_hi    : search window bounds
        all_peak_idxs     : list[int]   — all local-max indices in window
        all_peak_values   : list[float] — raw J values at each peak
        best_peak_idx     : int   — index of the highest peak (abs max if none)
        best_peak_value   : float — raw J at best peak
        best_peak_value_filt : float — filtered J at best peak
        best_lag          : int   — best_peak_idx − vel_peak_idx
        found_local_max   : bool  — True if ≥1 local max existed
    """
    if len(vel_ts) == 0:
        return {"error": "empty velocity time series"}
    if len(j_ts) == 0:
        return {"error": "empty J time series"}

    vel_peak_idx   = int(np.argmax(np.abs(vel_ts)))
    vel_peak_value = float(vel_ts[vel_peak_idx])

    n      = len(j_filt)
    win_lo = max(0, vel_peak_idx - half_window)
    win_hi = min(n, vel_peak_idx + half_window + 1)

    j_window      = j_filt[win_lo:win_hi]
    peaks_rel, _  = find_peaks(j_window, prominence=min_prominence)

    found_local_max   = len(peaks_rel) > 0
    all_peak_idxs     = [win_lo + int(r) for r in peaks_rel]
    all_peak_values   = [float(j_ts[idx]) for idx in all_peak_idxs]

    if found_local_max:
        best_rel      = peaks_rel[int(np.argmax(j_window[peaks_rel]))]
        best_peak_idx = win_lo + int(best_rel)
    else:
        best_peak_idx = win_lo + int(np.argmax(j_window))

    return {
        "vel_peak_idx":         vel_peak_idx,
        "vel_peak_value":       vel_peak_value,
        "win_lo":               win_lo,
        "win_hi":               win_hi,
        "all_peak_idxs":        all_peak_idxs,
        "all_peak_values":      all_peak_values,
        "best_peak_idx":        best_peak_idx,
        "best_peak_value":      float(j_ts[best_peak_idx]),
        "best_peak_value_filt": float(j_filt[best_peak_idx]),
        "best_lag":             best_peak_idx - vel_peak_idx,
        "found_local_max":      found_local_max,
    }


# ---------------------------------------------------------------------------
# Plotting — 4-panel: velocity | acceleration | firing rate | J
# J peaks are searched in the velocity window only; all peaks are marked.
# ---------------------------------------------------------------------------

def _plot_session(
    session_label: str,
    stim: int,
    vel_ts: np.ndarray,
    accel_ts: np.ndarray,
    fr_ts: np.ndarray,
    j_ts: np.ndarray,
    j_filt: np.ndarray,
    res: dict,
    output_dir: str,
    filter_label: str,
) -> str:
    """4-panel figure: velocity | acceleration | firing rate | J coupling.

    J peak search uses the velocity window only.  All found peaks are marked
    with small circles; the highest peak gets a star and lag annotation.
    Acceleration is shown for visual reference only.
    """
    def _lag_dir(lag: int) -> str:
        return "leads" if lag < 0 else ("lags" if lag > 0 else "simultaneous")

    v_idx    = res["vel_peak_idx"]
    win_lo   = res["win_lo"]
    win_hi   = res["win_hi"]
    best_idx = res["best_peak_idx"]
    best_lag = res["best_lag"]
    all_idxs = res["all_peak_idxs"]
    all_vals = res["all_peak_values"]

    # Acceleration peak (for display only)
    a_idx = int(np.argmax(np.abs(accel_ts))) if len(accel_ts) else 0

    found_local_max = res["found_local_max"]
    n_peaks = len(all_idxs)
    lag_dir = _lag_dir(best_lag)

    # Title status line — prominent PEAK DETECTED / NO PEAK banner
    if found_local_max:
        peak_status = f"✓ PEAK DETECTED  ({n_peaks} local max in window)"
        status_color = "darkgreen"
    else:
        peak_status = "✗ NO PEAK DETECTED  (showing abs max in window)"
        status_color = "firebrick"

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Session: {session_label}  |  Stim: {stim}  |  Filter: {filter_label}\n"
        f"Max-vel idx={v_idx}  |  {peak_status}\n"
        f"Best J idx={best_idx}  lag={best_lag:+d} bins ({lag_dir})",
        fontsize=9, fontweight="bold", color=status_color,
    )

    WIN_ALPHA = 0.12

    # ── Panel 1: Velocity ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(vel_ts, color="navy", linewidth=1.5, label="X velocity")
    ax.axvspan(win_lo, win_hi, alpha=WIN_ALPHA, color="deepskyblue",
               label="J search window")
    ax.axvline(v_idx, color="deepskyblue", linestyle="--", linewidth=2,
               label=f"Max vel (idx={v_idx})")
    ax.set_ylabel("X Velocity")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 2: Acceleration (reference only) ────────────────────────────
    ax = axes[1]
    ax.plot(accel_ts, color="steelblue", linewidth=1.5, label="X acceleration")
    ax.axvline(a_idx, color="limegreen", linestyle="--", linewidth=1.8,
               label=f"Max accel (idx={a_idx}, ref only)")
    ax.axvspan(win_lo, win_hi, alpha=WIN_ALPHA, color="deepskyblue",
               label="Vel J-search window")
    ax.set_ylabel("X Acceleration")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 3: Firing rate ──────────────────────────────────────────────
    ax = axes[2]
    if len(fr_ts) > 0:
        ax.plot(fr_ts, color="darkorange", linewidth=1.5, label="Firing rate")
        ax.axvline(v_idx, color="deepskyblue", linestyle="--", linewidth=1.2,
                   alpha=0.6, label=f"Max vel (idx={v_idx})")
        ax.axvline(a_idx, color="limegreen", linestyle="--", linewidth=1.2,
                   alpha=0.6, label=f"Max accel (idx={a_idx})")
    else:
        ax.text(0.5, 0.5, "Firing rate not available",
                ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_ylabel("Firing Rate")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 4: J coupling — all peaks in vel window marked ──────────────
    ax = axes[3]
    ax.plot(j_ts,   color="darkorchid", linewidth=1.0, alpha=0.35, label="J (raw)")
    ax.plot(j_filt, color="indigo",     linewidth=2.0, label="J (filtered)")
    ax.axvspan(win_lo, win_hi, alpha=WIN_ALPHA, color="deepskyblue",
               label="Vel search window")
    ax.axvline(v_idx, color="deepskyblue", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Max vel (idx={v_idx})")
    ax.axvline(a_idx, color="limegreen", linestyle="--", linewidth=1.2,
               alpha=0.5, label=f"Max accel (idx={a_idx}, ref)")

    # All detected local peaks — orange circles
    if all_idxs:
        filt_at_peaks = [float(j_filt[i]) for i in all_idxs]
        ax.scatter(all_idxs, filt_at_peaks,
                   color="orange", zorder=5, s=60, marker="o",
                   label=f"Local peaks in window ({n_peaks})")

    # Best point — appearance depends on whether a real peak was found
    if found_local_max:
        # Green star = genuine local maximum detected
        best_color  = "green"
        best_marker = "*"
        best_label  = (f"PEAK DETECTED idx={best_idx} "
                       f"lag={best_lag:+d}  J={res['best_peak_value']:.4f}")
    else:
        # Red X = no local max; this is just the highest point in the range
        best_color  = "firebrick"
        best_marker = "X"
        best_label  = (f"NO PEAK — abs max idx={best_idx} "
                       f"lag={best_lag:+d}  J={res['best_peak_value']:.4f}")

    ax.axvline(best_idx, color=best_color, linestyle="-.", linewidth=2.2,
               label=best_label)
    ax.scatter([best_idx], [res["best_peak_value"]],
               color=best_color, zorder=6, s=180, marker=best_marker)

    # Status text box in the J panel
    ax.text(0.01, 0.97, peak_status,
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=status_color, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=status_color, alpha=0.85))

    ax.set_ylabel("J (coupling)", color="darkorchid")
    ax.tick_params(axis="y", labelcolor="darkorchid")
    ax.set_xlabel("Time bin")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_label = session_label.replace("/", "_").replace("\\", "_")
    fname  = f"stim{stim}_{safe_label}.png"
    fpath  = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Session ID helpers (same convention as arbitration_j_many.py)
# ---------------------------------------------------------------------------

def _session_id_from_path(path: str) -> str:
    """Last 6 characters before '_results' in the path."""
    return path.split("_results")[0][-6:]


def _resolve_sessions(
    data_folder: str,
    requested_ids: list[str] | None,
    verbose_find: bool,
) -> dict[str, list[str]]:
    """
    Recursively find all per_reach_state.csv under *data_folder*.

    If *requested_ids* is None or empty every discovered session is returned.
    Otherwise only sessions whose ID appears in *requested_ids* are returned.

    Handles both the old layout (one CSV per session with all stims) and the
    new per-stim layout (one CSV per stim subdir, same session ID).

    Returns  dict  session_id -> [csv_path, ...]
    """
    all_csvs = find_file_recursive(
        data_folder, "per_reach_state.csv", verbose=verbose_find
    )

    # Accumulate all paths per session ID
    found_map: dict[str, list[str]] = {}
    for csv_path in all_csvs:
        try:
            sid = _session_id_from_path(csv_path)
        except (IndexError, ValueError):
            continue
        found_map.setdefault(sid, []).append(csv_path)

    # If no filter given, return everything
    if not requested_ids:
        print(f"[INFO] No --sessions filter given — processing all "
              f"{len(found_map)} session(s) found.")
        return dict(found_map)

    requested_set = set(requested_ids)
    matched: dict[str, list[str]] = {}
    for sid in requested_ids:
        if sid in found_map:
            matched[sid] = found_map[sid]
        else:
            print(f"[WARN] Session ID '{sid}' not found under {data_folder}",
                  file=sys.stderr)

    not_requested = set(found_map) - requested_set
    if not_requested:
        print(f"[INFO] {len(not_requested)} session(s) found but not requested.")

    return matched


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_sessions(
    matched: dict[str, list[str]],
    stim_range: range,
    half_window: int,
    cutoff: float,
    filter_order: int,
    use_gaussian: bool,
    smooth_sigma: float,
    min_prominence: float,
    output_dir: str,
    verbose: bool,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)

    if use_gaussian:
        filter_label = f"Gaussian σ={smooth_sigma}"
    elif cutoff > 0:
        filter_label = f"Butterworth LP cutoff={cutoff} order={filter_order}"
    else:
        filter_label = "No filter"

    def _dir(lag: int) -> str:
        return "leads" if lag < 0 else ("lags" if lag > 0 else "simultaneous")

    rows: list[dict] = []

    for session_id, csv_paths in matched.items():
        if verbose:
            sep = "─" * 100
            print(f"\n{sep}")
            print(f"  Session ID: {session_id}  ({len(csv_paths)} file(s))")
            print(sep)

        # Process each CSV file separately so model quality can be resolved
        # from its own directory (per-stim layout) or shared directory (old layout).
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Could not read {csv_path}: {e}", file=sys.stderr)
                continue

            csv_dir = os.path.dirname(csv_path)
            mq = _read_model_quality(csv_dir)

            if verbose:
                print(f"  File: {csv_path}")
                n_str = str(int(mq["n_neurons"])) if not np.isnan(mq["n_neurons"]) else "n/a"
                print(f"  N_neurons={n_str}  r_ising={mq['r_ising']:.4f}"
                      f"  r_independent={mq['r_independent']:.4f}"
                      if not np.isnan(mq["r_ising"])
                      else f"  N_neurons={n_str}  (model quality CSVs not found)")

            available_stims = sorted(df["stim"].unique()) if "stim" in df.columns else []
            stims_to_process = [s for s in stim_range if s in available_stims] or available_stims

            if verbose:
                hdr = (f"  {'Stim':>5}  {'VelPk':>7}  {'#Peaks':>7}  "
                       f"{'BestIdx':>8}  {'BestLag':>8}  {'Dir':>14}  {'PeakFound':>12}")
                print(hdr)
                print("  " + "-" * (len(hdr) - 2))

            for stim in stims_to_process:
                df_stim = df[df["stim"] == stim]
                if len(df_stim) == 0:
                    continue

                vel_ts   = _velocity_timeseries(df_stim)
                accel_ts = _acceleration_timeseries(df_stim)
                j_ts     = _mean_timeseries(df_stim, "j")
                fr_ts    = _mean_timeseries(df_stim, "firing_rate")

                if len(vel_ts) == 0 or len(j_ts) == 0:
                    if verbose:
                        print(f"  {stim:>5}  [no data]")
                    continue

                j_filt = _apply_filter(j_ts, cutoff, filter_order, use_gaussian, smooth_sigma)

                res = find_j_peaks_near_velocity(
                    vel_ts=vel_ts, j_ts=j_ts, j_filt=j_filt,
                    half_window=half_window, min_prominence=min_prominence,
                )

                if "error" in res:
                    if verbose:
                        print(f"  {stim:>5}  [error: {res['error']}]")
                    continue

                plot_path = _plot_session(
                    session_label=session_id,
                    stim=stim,
                    vel_ts=vel_ts,
                    accel_ts=accel_ts,
                    fr_ts=fr_ts,
                    j_ts=j_ts,
                    j_filt=j_filt,
                    res=res,
                    output_dir=output_dir,
                    filter_label=filter_label,
                )

                if verbose:
                    if res["found_local_max"]:
                        peak_col = "*** PEAK ***"
                    else:
                        peak_col = "  no peak   "
                    print(
                        f"  {stim:>5}  {res['vel_peak_idx']:>7d}  "
                        f"{len(res['all_peak_idxs']):>7d}  "
                        f"{res['best_peak_idx']:>8d}  {res['best_lag']:>+8d}  "
                        f"{_dir(res['best_lag']):>14}  {peak_col:>12}"
                    )

                rows.append({
                    "session_id":        session_id,
                    "csv_path":          csv_path,
                    "stim":              stim,
                    # Model metadata
                    "n_neurons":         mq["n_neurons"],
                    "r_ising":           mq["r_ising"],
                    "r_independent":     mq["r_independent"],
                    # Kinematics
                    "vel_peak_idx":      res["vel_peak_idx"],
                    "vel_peak_value":    res["vel_peak_value"],
                    "accel_peak_idx":    int(np.argmax(np.abs(accel_ts))) if len(accel_ts) else np.nan,
                    "win_lo":            res["win_lo"],
                    "win_hi":            res["win_hi"],
                    # All peaks
                    "n_j_peaks_in_window": len(res["all_peak_idxs"]),
                    "all_peak_idxs":     ";".join(str(i) for i in res["all_peak_idxs"]),
                    "all_peak_values":   ";".join(f"{v:.6f}" for v in res["all_peak_values"]),
                    # Highest peak
                    "has_j_peak":        res["found_local_max"],
                    "best_peak_idx":     res["best_peak_idx"],
                    "best_peak_value":   res["best_peak_value"],
                    "best_lag":          res["best_lag"],
                    "best_lag_direction": _dir(res["best_lag"]),
                    "plot_path":         plot_path,
                })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Find ALL J peaks inside the max-velocity window for manually supplied "
            "session IDs. Acceleration is shown for reference. Reports lag for the "
            "highest peak. Includes N_neurons and model-vs-independent fit in output CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    p.add_argument(
        "--data_folder", required=True, metavar="DIR",
        help="Root folder to search recursively for per_reach_state.csv files.",
    )
    p.add_argument(
        "--sessions", nargs="+", default=None, metavar="SESSION_ID",
        help=(
            "One or more session IDs to process (6-character IDs as they appear "
            "before '_results' in the path, e.g. 123456 789ABC). "
            "If omitted, every session found under --data_folder is processed."
        ),
    )

    # Stimulus filter
    p.add_argument("--stim_min", type=int, default=0,
                   help="Inclusive minimum stimulus index (default 0)")
    p.add_argument("--stim_max_exclusive", type=int, default=3,
                   help="Exclusive max stimulus index (default 3)")

    # Search window
    p.add_argument(
        "--half_window", type=int, default=60,
        help="Half-width (bins) of the search window around each kinematic peak (default 60).",
    )

    # Filtering
    filter_grp = p.add_argument_group("Low-pass filter (applied to J before peak detection)")
    filter_grp.add_argument(
        "--cutoff", type=float, default=0.2,
        help=(
            "Normalised cutoff frequency for Butterworth low-pass filter "
            "(0 = disabled; range (0, 1) where 1 = Nyquist). "
            "Higher = less smoothing. Default 0.2."
        ),
    )
    filter_grp.add_argument(
        "--filter_order", type=int, default=4,
        help="Butterworth filter order (default 4).",
    )
    filter_grp.add_argument(
        "--gaussian", action="store_true",
        help="Use Gaussian smoothing instead of Butterworth (see --smooth_sigma).",
    )
    filter_grp.add_argument(
        "--smooth_sigma", type=float, default=5.0,
        help="Gaussian smoothing sigma in time-bins (default 5; only when --gaussian).",
    )
    filter_grp.add_argument(
        "--min_prominence", type=float, default=0.0,
        help="Minimum peak prominence for scipy find_peaks (default 0).",
    )

    # Output
    p.add_argument(
        "--output_dir", type=str, default="./peak_near_max_velocity_results",
        help="Directory for plots and summary CSV.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-session table.")
    p.add_argument("--quiet_find", action="store_true",
                   help="Suppress verbose output from the recursive file search.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    if not os.path.isdir(data_folder):
        print(f"Error: {data_folder} is not a directory", file=sys.stderr)
        return 1

    stim_range = range(args.stim_min, args.stim_max_exclusive)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    matched = _resolve_sessions(
        data_folder=data_folder,
        requested_ids=args.sessions,
        verbose_find=not args.quiet_find,
    )

    if not matched:
        print("No matching sessions found. Check your session IDs and data folder.",
              file=sys.stderr)
        return 1

    n_requested = len(args.sessions) if args.sessions else len(matched)
    print(f"\nProcessing {len(matched)}/{n_requested} session(s):")
    for sid, paths in matched.items():
        for path in paths:
            print(f"  {sid}  →  {path}")

    rows = run_sessions(
        matched=matched,
        stim_range=stim_range,
        half_window=args.half_window,
        cutoff=args.cutoff,
        filter_order=args.filter_order,
        use_gaussian=args.gaussian,
        smooth_sigma=args.smooth_sigma,
        min_prominence=args.min_prominence,
        output_dir=output_dir,
        verbose=not args.quiet,
    )

    if not rows:
        print("No results produced.", file=sys.stderr)
        return 1

    summary_df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, "peak_near_max_velocity_summary.csv")
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary CSV written to: {csv_out}")
    print(f"Plots saved in:         {output_dir}")

    n_total    = len(summary_df)
    n_has_peak = summary_df["has_j_peak"].sum()
    n_leads    = (summary_df["best_lag_direction"] == "leads").sum()
    n_lags     = (summary_df["best_lag_direction"] == "lags").sum()
    n_sim      = (summary_df["best_lag_direction"] == "simultaneous").sum()
    mean_lag   = summary_df["best_lag"].mean()

    print(f"\nOverall ({n_total} session×stim pairs):")
    print(f"  Has local J peak in vel window : {n_has_peak}/{n_total}")
    print(f"  J leads vel peak               : {n_leads}")
    print(f"  J lags  vel peak               : {n_lags}")
    print(f"  Simultaneous                   : {n_sim}")
    print(f"  Mean lag (best peak, bins)     : {mean_lag:+.1f}")

    if "r_ising" in summary_df.columns:
        mean_ri = summary_df["r_ising"].mean()
        mean_rind = summary_df["r_independent"].mean()
        print(f"  Mean r(P_data, P_ising)        : {mean_ri:.4f}")
        print(f"  Mean r(P_data, P_independent)  : {mean_rind:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
