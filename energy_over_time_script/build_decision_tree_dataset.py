#!/usr/bin/env python
# coding: utf-8
"""
Build a per-session/per-stimulus CSV dataset for decision-tree analysis.

Each row represents one (session, stim) pair.  The target variable is
`is_j_peak` (did the J-coupling show a significant event near the kinematic
window?).  Features capture model quality, kinematics, and neural statistics
that might explain why some sessions show a J peak and others do not.

Columns written
---------------
session_id            : 6-digit session identifier
stim_number           : stimulus index
n_neurons             : number of neurons in this session

--- J peak detection (target + soft score) ---
is_j_peak             : 1 if a significant J event was detected, else 0
idx_j_peak            : absolute time-bin of detected J peak (NaN if none)
j_peak_ratio          : detection strength (max of prominence ratio and
                        level-shift z-score); higher = more exceptional

--- Kinematic peaks (features) ---
idx_velocity_max      : absolute time-bin of max x-velocity in window
vel_peak_value        : max x-velocity in window
idx_acceleration_max  : absolute time-bin of max |x-acceleration| in window
accel_peak_value      : max |x-acceleration| in window

--- Post-peak kinematic noise (features) ---
var_vel_after_max     : variance of x-velocity from vel peak to window end
var_accel_after_max   : variance of x-acceleration from accel peak to window end

--- Model quality (features) ---
r_ising               : Pearson r(P_data, P_ising)     — how well Ising fits data
r_independent         : Pearson r(P_data, P_independent) — how well indep. fits data
ising_indep_dist      : sum |P_ising(k) - P_independent(k)| — total-variation
                        distance between the two model distributions
                        (large = Ising and independent predict very different
                        population activity; small = they are nearly equivalent)

--- J statistics in window (features) ---
mean_j_in_window      : mean J coupling in the search window
std_j_in_window       : std  J coupling in the search window

Usage
-----
    python build_decision_tree_dataset.py \\
        --data_folder /path/to/energy_decomp/ \\
        --window 350 475 \\
        --stim_min 0 --stim_max_exclusive 3 \\
        --output ./decision_tree_dataset.csv

Optional filters (same as arbitration_j_many.py):
    --sessions 210425 210511 ...   (restrict to specific sessions)
    --rep_start 1 --rep_end_exclusive 2
    --peak_threshold 1.75          (detection threshold)
    --smooth_sigma 5.0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from multiprocessing import Pool

from src.util import find_file_recursive
from src.processing import calculate_session_averages
from src.peak_detection import detect_j_peak, gaussian_smooth


# ---------------------------------------------------------------------------
# Helpers shared with arbitration_j_many.py
# ---------------------------------------------------------------------------

def _session_id_from_path(session: str) -> str:
    return session.split("_results")[0][-6:]


def _rep_from_session_path(session: str) -> str:
    return session.split("_results/")[1].split("/")[0].split("_")[1][3:]


def _mean_ts(df: pd.DataFrame, col: str) -> np.ndarray:
    """Mean time-series of *col* across all reaches in a session DataFrame."""
    parts = []
    for _, grp in df.groupby("reach_idx"):
        parts.append(grp[col].values)
    if not parts:
        return np.array([])
    n = min(len(p) for p in parts)
    return np.mean([p[:n] for p in parts], axis=0)


def _velocity_ts(df: pd.DataFrame) -> np.ndarray:
    """Mean x-velocity time series (finite-difference of x position)."""
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
# Model quality reader
# ---------------------------------------------------------------------------

def _read_model_quality(stim_dir: str) -> dict:
    """
    Read model_quality_summary_P_K.csv and _P_K_metadata.csv from *stim_dir*.

    Returns
    -------
    dict with keys:
        n_neurons        : int   (NaN if unavailable)
        r_ising          : float Pearson r(P_data, P_ising)
        r_independent    : float Pearson r(P_data, P_independent)
        ising_indep_dist : float sum |P_ising - P_independent|
    """
    result = {
        "n_neurons":          np.nan,
        "r_ising":            np.nan,
        "r_independent":      np.nan,
        "r_ising_vs_indep":   np.nan,   # corr(P_ising, P_independent) — how similar the two models are
        "ising_indep_dist":   np.nan,
    }

    pk_path   = os.path.join(stim_dir, "model_quality_summary_P_K.csv")
    meta_path = os.path.join(stim_dir, "model_quality_summary_P_K_metadata.csv")

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
                p_data  = pk["P_data"].values.astype(float)
                p_ising = pk["P_ising"].values.astype(float)
                p_indep = pk["P_independent"].values.astype(float)
                result["r_ising"]          = float(np.corrcoef(p_data, p_ising)[0, 1])
                result["r_independent"]    = float(np.corrcoef(p_data, p_indep)[0, 1])
                result["r_ising_vs_indep"] = float(np.corrcoef(p_ising, p_indep)[0, 1])
                result["ising_indep_dist"] = float(np.sum(np.abs(p_ising - p_indep)))
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Per-session feature extraction
# ---------------------------------------------------------------------------

def _extract_features(session_path: str, session_id: str, stim: int,
                       df: pd.DataFrame, w_lo: int, w_hi: int,
                       peak_threshold: float, smooth_sigma: float) -> dict | None:
    """
    Extract all features for one (session, stim) pair.

    Returns a dict of column → value, or None if extraction fails.
    """
    try:
        # In the new per-stim layout the CSV already contains only one stim;
        # in the old layout we filter by stim column.
        stim_df = df[df["stim"] == stim] if "stim" in df.columns else df
        if stim_df.empty:
            return None

        # ── Time series ─────────────────────────────────────────────────
        vel_ts   = _velocity_ts(stim_df)
        accel_ts = _acceleration_ts(stim_df)
        j_ts     = _mean_ts(stim_df, "j")

        if len(vel_ts) == 0 or len(j_ts) == 0:
            return None

        n = len(vel_ts)
        lo = max(0, w_lo)
        hi = min(n, w_hi)

        # ── Kinematic peaks in window ────────────────────────────────────
        vel_win   = vel_ts[lo:hi]
        accel_win = accel_ts[lo:hi] if len(accel_ts) >= hi else np.array([])

        if len(vel_win) == 0:
            return None

        vel_rel_idx   = int(np.argmax(np.abs(vel_win)))
        vel_peak_idx  = lo + vel_rel_idx
        vel_peak_val  = float(vel_win[vel_rel_idx])

        if len(accel_win) > 0:
            accel_rel_idx  = int(np.argmax(np.abs(accel_win)))
            accel_peak_idx = lo + accel_rel_idx
            accel_peak_val = float(accel_win[accel_rel_idx])
        else:
            accel_peak_idx = np.nan
            accel_peak_val = np.nan

        # ── Post-peak kinematic variance (noisiness from peak to w_hi) ──
        var_vel_after   = float(np.var(vel_ts[vel_peak_idx:hi]))   if vel_peak_idx < hi else np.nan
        if not np.isnan(accel_peak_idx) and int(accel_peak_idx) < hi:
            var_accel_after = float(np.var(accel_ts[int(accel_peak_idx):hi]))
        else:
            var_accel_after = np.nan

        # ── J peak detection ─────────────────────────────────────────────
        has_peak, peak_idx, _, peak_ratio = detect_j_peak(
            j_ts, lo, hi,
            threshold_ratio=peak_threshold,
            smooth_sigma=smooth_sigma,
        )

        # ── J statistics in window ───────────────────────────────────────
        j_win = j_ts[lo:hi]
        mean_j = float(np.mean(j_win)) if len(j_win) else np.nan
        std_j  = float(np.std(j_win))  if len(j_win) else np.nan

        # ── Model quality (same directory as per_reach_state.csv) ────────
        # All stims share one Ising fit; model_quality_summary_P_K.csv is
        # at the session level, not in a per-stim subfolder.
        session_dir = os.path.dirname(session_path)
        mq = _read_model_quality(session_dir)

        return {
            "session_id":          session_id,
            "stim_number":         stim,
            "n_neurons":           mq["n_neurons"],
            # Target
            "is_j_peak":           int(has_peak),
            "idx_j_peak":          int(peak_idx) if has_peak else np.nan,
            "j_peak_ratio":        float(peak_ratio) if not np.isnan(peak_ratio) else np.nan,
            # Kinematic peaks
            "idx_velocity_max":    vel_peak_idx,
            "vel_peak_value":      vel_peak_val,
            "idx_acceleration_max": int(accel_peak_idx) if not np.isnan(accel_peak_idx) else np.nan,
            "accel_peak_value":    accel_peak_val,
            # Post-peak noise
            "var_vel_after_max":   var_vel_after,
            "var_accel_after_max": var_accel_after,
            # Model quality
            "r_ising":             mq["r_ising"],
            "r_independent":       mq["r_independent"],
            "r_ising_vs_indep":    mq["r_ising_vs_indep"],
            "ising_indep_dist":    mq["ising_indep_dist"],
            # J statistics
            "mean_j_in_window":    mean_j,
            "std_j_in_window":     std_j,
        }

    except Exception as e:
        print(f"  [WARN] {session_id} stim={stim}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _stim_from_path(path: str) -> int | None:
    """
    If *path* contains a ``stim_N`` directory component, return N.
    Returns None if no stim folder is found (old per-session format).
    """
    import re
    for part in path.replace("\\", "/").split("/"):
        m = re.fullmatch(r"stim_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def _session_id_from_path_new(path: str) -> str:
    """
    Extract a 6-digit session ID (YYMMDD date) from a path that uses the
    new per-stim layout.

    Strategy: walk the path components from the stim_N component upward,
    returning the first 6-digit sequence found.  This handles both:
      .../210421_results/210421_rep1/full_reach/stim_0/per_reach_state.csv
      .../210421_stim_0/per_reach_state.csv
    """
    import re
    parts = path.replace("\\", "/").split("/")
    stim_idx = next(
        (i for i, p in enumerate(parts) if re.fullmatch(r"stim_\d+", p)), None)

    # Search from just above stim_N all the way to the root
    search_parts = parts[:stim_idx] if stim_idx is not None else parts
    for part in reversed(search_parts):   # nearest ancestor first
        m = re.search(r"(\d{6})", part)
        if m:
            return m.group(1)
    return "unknown"


# ---------------------------------------------------------------------------
# Multiprocessing task wrapper
# ---------------------------------------------------------------------------

def _task(args):
    session_path, session_id, stim, df, w_lo, w_hi, peak_threshold, smooth_sigma = args
    row = _extract_features(
        session_path, session_id, stim, df, w_lo, w_hi, peak_threshold, smooth_sigma)
    print(f"  ✓ {session_id}  stim={stim}  is_j_peak={row['is_j_peak'] if row else 'ERR'}")
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a per-session CSV dataset for decision-tree analysis of J peaks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_folder", required=True,
                   help="Root folder containing energy_decomp session subdirectories "
                        "(same as --data_folder in arbitration_j_many.py).")
    p.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"),
                   default=[350, 475],
                   help="Search window [LO, HI) in time bins.")
    p.add_argument("--stim_min", type=int, default=0,
                   help="First stimulus index (inclusive).")
    p.add_argument("--stim_max_exclusive", type=int, default=3,
                   help="Last stimulus index (exclusive).")
    p.add_argument("--sessions", nargs="*", default=None,
                   help="Restrict to these session IDs (e.g. 210425 220516).")
    p.add_argument("--rep_start", type=int, default=1,
                   help="First rep to include.")
    p.add_argument("--rep_end_exclusive", type=int, default=2,
                   help="Last rep (exclusive).")
    p.add_argument("--peak_threshold", type=float, default=1.75,
                   help="Prominence/level-shift threshold for J peak detection.")
    p.add_argument("--smooth_sigma", type=float, default=5.0,
                   help="Gaussian smoothing sigma for peak detection (bins).")
    p.add_argument("--output", default="./decision_tree_dataset.csv",
                   help="Output CSV path.")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel worker processes (default: all CPUs).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress file-search progress messages.")
    return p.parse_args()


def main():
    args = parse_args()

    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    if not os.path.isdir(data_folder):
        print(f"Error: {data_folder} is not a directory", file=sys.stderr)
        return 1

    w_lo, w_hi    = args.window
    stim_range    = range(args.stim_min, args.stim_max_exclusive)
    session_whitelist = set(args.sessions) if args.sessions else None

    print(f"Scanning for per_reach_state.csv under {data_folder} …")
    all_reach_states = find_file_recursive(
        data_folder, "per_reach_state.csv", verbose=not args.quiet)
    if not all_reach_states:
        print(f"No per_reach_state.csv found under {data_folder}", file=sys.stderr)
        return 1

    if session_whitelist:
        print(f"Session filter active — keeping: {sorted(session_whitelist)}")

    rows = []

    # ── Detect layout ──────────────────────────────────────────────────────
    # New layout: per_reach_state.csv lives inside a stim_N/ subfolder.
    # Old layout: per_reach_state.csv is at the session level (all stims in one file).
    new_format_paths = [p for p in all_reach_states if _stim_from_path(p) is not None]
    old_format_paths = [p for p in all_reach_states if _stim_from_path(p) is None]

    tasks = []

    # ── New per-stim layout ────────────────────────────────────────────────
    if new_format_paths:
        print(f"\nNew per-stim layout: {len(new_format_paths)} stim CSVs found.")
        for path in new_format_paths:
            stim = _stim_from_path(path)
            if stim not in stim_range:
                continue
            sid = _session_id_from_path_new(path)
            if session_whitelist and sid not in session_whitelist:
                continue
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  [WARN] Could not read {path}: {e}", file=sys.stderr)
                continue
            tasks.append((path, sid, stim, df, w_lo, w_hi,
                          args.peak_threshold, args.smooth_sigma))

    # ── Old per-session layout ─────────────────────────────────────────────
    if old_format_paths:
        for rep in range(args.rep_start, args.rep_end_exclusive):
            rep_s = str(rep)
            session_paths: dict[str, str] = {}

            for session_path in old_format_paths:
                if "full_reach" not in session_path:
                    continue
                try:
                    if _rep_from_session_path(session_path) != rep_s:
                        continue
                except (IndexError, ValueError):
                    continue

                sid = _session_id_from_path(session_path)
                if session_whitelist and sid not in session_whitelist:
                    continue
                session_paths[sid] = session_path

            if not session_paths:
                continue

            print(f"\nOld layout — rep {rep}: {len(session_paths)} sessions "
                  f"× {len(stim_range)} stims")

            for sid, path in session_paths.items():
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print(f"  [WARN] Could not read {path}: {e}", file=sys.stderr)
                    continue
                for stim in stim_range:
                    tasks.append((path, sid, stim, df, w_lo, w_hi,
                                  args.peak_threshold, args.smooth_sigma))

    if not tasks:
        print("No tasks generated — check --data_folder and filters.", file=sys.stderr)
        return 1

    print(f"\nTotal tasks: {len(tasks)}")
    pool_kw = {} if args.workers is None else {"processes": args.workers}
    with Pool(**pool_kw) as pool:
        for row in pool.imap_unordered(_task, tasks):
            if row is not None:
                rows.append(row)

    if not rows:
        print("No rows generated — nothing to write.", file=sys.stderr)
        return 1

    df_out = pd.DataFrame(rows, columns=[
        "session_id", "stim_number", "n_neurons",
        "is_j_peak", "idx_j_peak", "j_peak_ratio",
        "idx_velocity_max", "vel_peak_value",
        "idx_acceleration_max", "accel_peak_value",
        "var_vel_after_max", "var_accel_after_max",
        "r_ising", "r_independent", "r_ising_vs_indep", "ising_indep_dist",
        "mean_j_in_window", "std_j_in_window",
    ])
    df_out.sort_values(["session_id", "stim_number"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    out_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out.to_csv(out_path, index=False)

    n_peaks = int(df_out["is_j_peak"].sum())
    print(f"\nWrote {len(df_out)} rows → {out_path}")
    print(f"J peak present: {n_peaks}/{len(df_out)} "
          f"({100*n_peaks/len(df_out):.1f}%)")
    print("\nColumn summary:")
    print(df_out.describe(include="all").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
