#!/usr/bin/env python
# coding: utf-8
"""
Find the highest J peak near max-velocity AND max-acceleration for a manually
supplied list of session IDs, discovered by recursively searching a data folder.

Workflow per session / stimulus:
  1. Recurse through --data_folder to find all per_reach_state.csv files.
  2. Match each file against the session IDs supplied via --sessions
     (last 6 characters before "_results" in the path, same convention as
     arbitration_j_many.py).  Unrecognised IDs are reported and skipped.
  3. Compute mean x-velocity, x-acceleration, and firing rate across reaches.
  4. Locate max-velocity and max-acceleration indices separately.
  5. Apply an optional low-pass (Butterworth) filter to the J time-series.
  6. Within ±half_window bins around each kinematic peak, find the highest
     local maximum of J.  Falls back to the absolute max if no local max exists.
  7. Report timing lags: j_peak_idx − max_vel_idx  and  j_peak_idx − max_accel_idx.
  8. Save a four-panel plot (velocity | acceleration | firing rate | J) and a CSV.

Usage
-----
    python peak_near_max_velocity.py \\
        --data_folder /path/to/results_root \\
        --sessions 123456 789ABC DEF012 \\
        --stim_min 0 --stim_max_exclusive 3 \\
        --half_window 60 \\
        --cutoff 0.08 --filter_order 4 \\
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
# Core analysis — generic: find J peak near any kinematic reference peak
# ---------------------------------------------------------------------------

def find_j_peak_near_kinematic(
    ref_ts: np.ndarray,
    j_ts: np.ndarray,
    j_filt: np.ndarray,
    half_window: int = 60,
    min_prominence: float = 0.0,
) -> dict:
    """
    Find the highest J peak near the maximum of *ref_ts* (velocity or acceleration).

    Parameters
    ----------
    ref_ts        : kinematic reference time series (velocity or acceleration)
    j_ts          : raw J (coupling) time series
    j_filt        : filtered J time series (pre-computed, same length as j_ts)
    half_window   : search extends ±half_window bins from the kinematic peak
    min_prominence: minimum peak prominence for scipy find_peaks

    Returns
    -------
    dict with keys:
        ref_peak_idx    : index of the kinematic maximum (|ref_ts|)
        ref_peak_value  : value of ref_ts at that index
        win_lo, win_hi  : search window bounds (clipped to signal length)
        j_peak_idx      : index of the highest J local max in the window
        j_peak_value    : raw J value at j_peak_idx
        j_peak_value_filt : filtered J value at j_peak_idx
        lag             : j_peak_idx − ref_peak_idx
        found_local_max : True if a local maximum existed in the window
    """
    if len(ref_ts) == 0:
        return {"error": "empty reference time series"}
    if len(j_ts) == 0:
        return {"error": "empty J time series"}

    ref_peak_idx   = int(np.argmax(np.abs(ref_ts)))
    ref_peak_value = float(ref_ts[ref_peak_idx])

    n      = len(j_filt)
    win_lo = max(0, ref_peak_idx - half_window)
    win_hi = min(n, ref_peak_idx + half_window + 1)

    j_window  = j_filt[win_lo:win_hi]
    peaks_rel, _ = find_peaks(j_window, prominence=min_prominence)

    found_local_max = len(peaks_rel) > 0
    if found_local_max:
        best_rel   = peaks_rel[int(np.argmax(j_window[peaks_rel]))]
        j_peak_idx = win_lo + best_rel
    else:
        j_peak_idx = win_lo + int(np.argmax(j_window))

    return {
        "ref_peak_idx":      ref_peak_idx,
        "ref_peak_value":    ref_peak_value,
        "win_lo":            win_lo,
        "win_hi":            win_hi,
        "j_peak_idx":        int(j_peak_idx),
        "j_peak_value":      float(j_ts[j_peak_idx]),
        "j_peak_value_filt": float(j_filt[j_peak_idx]),
        "lag":               int(j_peak_idx) - ref_peak_idx,
        "found_local_max":   found_local_max,
    }


# ---------------------------------------------------------------------------
# Plotting — 4-panel: velocity | acceleration | firing rate | J
# ---------------------------------------------------------------------------

def _plot_session(
    session_label: str,
    stim: int,
    vel_ts: np.ndarray,
    accel_ts: np.ndarray,
    fr_ts: np.ndarray,
    j_ts: np.ndarray,
    j_filt: np.ndarray,
    res_vel: dict,
    res_accel: dict,
    output_dir: str,
    filter_label: str,
) -> str:
    """Four-panel figure: velocity | acceleration | firing rate | J coupling."""

    def _lag_dir(lag: int) -> str:
        return "leads" if lag < 0 else ("lags" if lag > 0 else "simultaneous")

    v_idx   = res_vel["ref_peak_idx"]
    a_idx   = res_accel["ref_peak_idx"]
    vj_idx  = res_vel["j_peak_idx"]
    aj_idx  = res_accel["j_peak_idx"]
    vj_lag  = res_vel["lag"]
    aj_lag  = res_accel["lag"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"Session: {session_label}  |  Stim: {stim}  |  Filter: {filter_label}\n"
        f"Vel peak idx={v_idx}  →  J peak idx={vj_idx}  lag={vj_lag:+d} bins "
        f"({_lag_dir(vj_lag)})\n"
        f"Accel peak idx={a_idx}  →  J peak idx={aj_idx}  lag={aj_lag:+d} bins "
        f"({_lag_dir(aj_lag)})",
        fontsize=9, fontweight="bold",
    )

    WINDOW_ALPHA = 0.12

    # ── Panel 1: Velocity ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(vel_ts, color="navy", linewidth=1.5, label="X velocity")
    ax.axvline(v_idx, color="deepskyblue", linestyle="--", linewidth=2,
               label=f"Max vel (idx={v_idx})")
    ax.axvspan(res_vel["win_lo"], res_vel["win_hi"],
               alpha=WINDOW_ALPHA, color="deepskyblue", label="Vel search window")
    ax.set_ylabel("X Velocity")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 2: Acceleration ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(accel_ts, color="steelblue", linewidth=1.5, label="X acceleration")
    ax.axvline(a_idx, color="limegreen", linestyle="--", linewidth=2,
               label=f"Max accel (idx={a_idx})")
    ax.axvspan(res_accel["win_lo"], res_accel["win_hi"],
               alpha=WINDOW_ALPHA, color="limegreen", label="Accel search window")
    ax.set_ylabel("X Acceleration")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ── Panel 3: Firing rate ──────────────────────────────────────────────
    ax = axes[2]
    if len(fr_ts) > 0:
        ax.plot(fr_ts, color="darkorange", linewidth=1.5, label="Firing rate")
        # Reference lines from both kinematics for easy visual comparison
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

    # ── Panel 4: J coupling ───────────────────────────────────────────────
    ax = axes[3]
    ax.plot(j_ts, color="darkorchid", linewidth=1.0, alpha=0.35, label="J (raw)")
    ax.plot(j_filt, color="indigo", linewidth=2.0, label="J (filtered)")

    # Velocity-anchored search window + J peak
    ax.axvspan(res_vel["win_lo"], res_vel["win_hi"],
               alpha=WINDOW_ALPHA, color="deepskyblue")
    ax.axvline(v_idx, color="deepskyblue", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Max vel (idx={v_idx})")
    ax.axvline(vj_idx, color="deepskyblue", linestyle="-.", linewidth=2.2,
               label=f"J peak@vel win (idx={vj_idx}, lag={vj_lag:+d})")
    ax.scatter([vj_idx], [res_vel["j_peak_value"]], color="deepskyblue",
               zorder=6, s=120, marker="*")

    # Acceleration-anchored search window + J peak
    ax.axvspan(res_accel["win_lo"], res_accel["win_hi"],
               alpha=WINDOW_ALPHA, color="limegreen")
    ax.axvline(a_idx, color="limegreen", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Max accel (idx={a_idx})")
    ax.axvline(aj_idx, color="limegreen", linestyle="-.", linewidth=2.2,
               label=f"J peak@accel win (idx={aj_idx}, lag={aj_lag:+d})")
    ax.scatter([aj_idx], [res_accel["j_peak_value"]], color="limegreen",
               zorder=6, s=120, marker="^")

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
    requested_ids: list[str],
    verbose_find: bool,
) -> dict[str, str]:
    """
    Recursively find all per_reach_state.csv under *data_folder* and return
    only those whose session ID appears in *requested_ids*.

    Returns  dict  session_id -> csv_path
    """
    all_csvs = find_file_recursive(
        data_folder, "per_reach_state.csv", verbose=verbose_find
    )

    found_map: dict[str, str] = {}
    for csv_path in all_csvs:
        try:
            sid = _session_id_from_path(csv_path)
        except (IndexError, ValueError):
            continue
        found_map[sid] = csv_path

    requested_set = set(requested_ids)
    matched: dict[str, str] = {}
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
    matched: dict[str, str],
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

    rows: list[dict] = []

    for session_id, csv_path in matched.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}", file=sys.stderr)
            continue

        if verbose:
            sep = "─" * 100
            print(f"\n{sep}")
            print(f"  Session ID: {session_id}")
            print(f"  File:       {csv_path}")
            print(sep)

        available_stims = sorted(df["stim"].unique()) if "stim" in df.columns else []
        stims_to_process = [s for s in stim_range if s in available_stims] or available_stims

        if verbose:
            hdr = (
                f"  {'Stim':>5}  "
                f"{'VelPk':>7}  {'JpkV':>6}  {'LagV':>6}  {'DirV':>14}  {'LMV':>5}  "
                f"{'AccPk':>7}  {'JpkA':>6}  {'LagA':>6}  {'DirA':>14}  {'LMA':>5}"
            )
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

            if len(vel_ts) == 0 or len(accel_ts) == 0 or len(j_ts) == 0:
                if verbose:
                    print(f"  {stim:>5}  [no data]")
                continue

            # Filter J once, reuse for both kinematic references
            j_filt = _apply_filter(j_ts, cutoff, filter_order, use_gaussian, smooth_sigma)

            res_vel = find_j_peak_near_kinematic(
                ref_ts=vel_ts, j_ts=j_ts, j_filt=j_filt,
                half_window=half_window, min_prominence=min_prominence,
            )
            res_accel = find_j_peak_near_kinematic(
                ref_ts=accel_ts, j_ts=j_ts, j_filt=j_filt,
                half_window=half_window, min_prominence=min_prominence,
            )

            if "error" in res_vel or "error" in res_accel:
                errs = "; ".join(filter(None, [res_vel.get("error"), res_accel.get("error")]))
                if verbose:
                    print(f"  {stim:>5}  [error: {errs}]")
                continue

            def _dir(lag: int) -> str:
                return "leads" if lag < 0 else ("lags" if lag > 0 else "simultaneous")

            plot_path = _plot_session(
                session_label=session_id,
                stim=stim,
                vel_ts=vel_ts,
                accel_ts=accel_ts,
                fr_ts=fr_ts,
                j_ts=j_ts,
                j_filt=j_filt,
                res_vel=res_vel,
                res_accel=res_accel,
                output_dir=output_dir,
                filter_label=filter_label,
            )

            if verbose:
                lmv = "yes" if res_vel["found_local_max"]   else "no"
                lma = "yes" if res_accel["found_local_max"] else "no"
                print(
                    f"  {stim:>5}  "
                    f"{res_vel['ref_peak_idx']:>7d}  {res_vel['j_peak_idx']:>6d}  "
                    f"{res_vel['lag']:>+6d}  {_dir(res_vel['lag']):>14}  {lmv:>5}  "
                    f"{res_accel['ref_peak_idx']:>7d}  {res_accel['j_peak_idx']:>6d}  "
                    f"{res_accel['lag']:>+6d}  {_dir(res_accel['lag']):>14}  {lma:>5}"
                )

            rows.append({
                "session_id":            session_id,
                "csv_path":              csv_path,
                "stim":                  stim,
                # Velocity-anchored results
                "vel_peak_idx":          res_vel["ref_peak_idx"],
                "vel_peak_value":        res_vel["ref_peak_value"],
                "vel_win_lo":            res_vel["win_lo"],
                "vel_win_hi":            res_vel["win_hi"],
                "j_peak_idx_vel":        res_vel["j_peak_idx"],
                "j_peak_value_vel":      res_vel["j_peak_value"],
                "j_peak_value_filt_vel": res_vel["j_peak_value_filt"],
                "lag_vel":               res_vel["lag"],
                "direction_vel":         _dir(res_vel["lag"]),
                "found_local_max_vel":   res_vel["found_local_max"],
                # Acceleration-anchored results
                "accel_peak_idx":        res_accel["ref_peak_idx"],
                "accel_peak_value":      res_accel["ref_peak_value"],
                "accel_win_lo":          res_accel["win_lo"],
                "accel_win_hi":          res_accel["win_hi"],
                "j_peak_idx_accel":      res_accel["j_peak_idx"],
                "j_peak_value_accel":    res_accel["j_peak_value"],
                "j_peak_value_filt_accel": res_accel["j_peak_value_filt"],
                "lag_accel":             res_accel["lag"],
                "direction_accel":       _dir(res_accel["lag"]),
                "found_local_max_accel": res_accel["found_local_max"],
                "plot_path":             plot_path,
            })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Find the highest J peak near max velocity AND max acceleration, "
            "for a manually supplied list of session IDs discovered by recursively "
            "searching a data folder."
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
        "--sessions", nargs="+", required=True, metavar="SESSION_ID",
        help=(
            "One or more session IDs to process (6-character IDs as they appear "
            "before '_results' in the path, e.g. 123456 789ABC)."
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

    print(f"\nMatched {len(matched)}/{len(args.sessions)} requested session(s):")
    for sid, path in matched.items():
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

    # Brief overall stats for both kinematic references
    for ref, col in [("velocity", "vel"), ("acceleration", "accel")]:
        n_leads  = (summary_df[f"direction_{col}"] == "leads").sum()
        n_lags   = (summary_df[f"direction_{col}"] == "lags").sum()
        n_sim    = (summary_df[f"direction_{col}"] == "simultaneous").sum()
        mean_lag = summary_df[f"lag_{col}"].mean()
        print(f"\nVs {ref} ({len(summary_df)} session×stim pairs):")
        print(f"  J leads  : {n_leads}")
        print(f"  J lags   : {n_lags}")
        print(f"  Simult.  : {n_sim}")
        print(f"  Mean lag : {mean_lag:+.1f} bins")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
