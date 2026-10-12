"""
test_j_timing_vs_kinematics.py
-------------------------------
Is the J-coupling peak's timing closer to acceleration, or to firing rate?

For every session/stim, the candidate J-peak index (the tallest local
maximum of J within the analysis window -- the same candidate detect_j_peak
scores, kept regardless of whether it cleared the significance threshold)
is compared against:
  - nearest of accel_max_idx / accel_min_idx : index of the raw max OR min
    (signed) acceleration in the window, whichever is closer to the J index
  - nearest of fr_peak_idx / fr_trough_idx   : index of the max OR min
    firing rate in the window, whichever is closer

Both sides use "closer of the min or max" rather than collapsing to a
single abs-value peak, so acceleration and firing rate are compared on
the same footing.

A paired Wilcoxon signed-rank test on |j_idx - accel_idx| vs.
|j_idx - fr_idx| is run three ways:
  1. J-peak sessions only      (has_j_peak == True)
  2. No/weak-peak sessions only (has_j_peak == False)
  3. All sessions, regardless of whether the J signal actually had a
     significant peak -- the raw candidate index is used either way.

Usage
-----
  python test_j_timing_vs_kinematics.py --data_folder outputs/energy_decomp_Jul_10_svg
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.util import find_file_recursive
from src.peak_detection import detect_j_peak, detect_signal_extremum
from pk_j_analysis import (
    load_session_pk_series, _stim_from_path, _session_id_from_path,
)


def _raw_candidate_indices(data: dict, peak_threshold: float, smooth_sigma: float):
    """
    Re-run peak detection on the already-loaded J/firing-rate series to
    recover the *raw* candidate index regardless of significance --
    load_session_pk_series() nulls out j_peak_idx/fr_peak_idx/fr_trough_idx
    whenever has_j_peak/has_fr_peak/has_fr_trough is False, which is exactly
    the information this analysis needs back.
    """
    w_lo, w_hi = data["w_lo"], data["w_hi"]

    _, j_idx, _, j_ratio = detect_j_peak(
        data["j_ts"], w_lo, w_hi,
        threshold_ratio=peak_threshold, smooth_sigma=smooth_sigma)

    fr_ts = data["fr_ts"]
    if len(fr_ts) > 0:
        _, fr_peak_idx, _, _ = detect_signal_extremum(
            fr_ts, w_lo, w_hi, kind="peak", smooth_sigma=smooth_sigma)
        _, fr_trough_idx, _, _ = detect_signal_extremum(
            fr_ts, w_lo, w_hi, kind="trough", smooth_sigma=smooth_sigma)
    else:
        fr_peak_idx = fr_trough_idx = None

    return j_idx, j_ratio, fr_peak_idx, fr_trough_idx


def _plot_verification(session_id, stim, data, j_idx, accel_max_idx, accel_min_idx,
                       fr_peak_idx, fr_trough_idx, dist_accel, dist_fr, closer_to,
                       save_path):
    """
    3-panel figure (J, acceleration, firing rate) over the FULL recording
    (not just the analysis window), with the window shaded so the candidate
    markers can be seen in context -- e.g. whether a signal was already
    trending toward a bigger extremum just outside the window, which a
    window-only view would crop out and make look misleadingly clean.
    """
    w_lo, w_hi = data["w_lo"], data["w_hi"]
    n_max = max(len(data["j_ts"]), len(data["accel_ts"]), len(data["fr_ts"]))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, layout="constrained")

    def _shade(ax):
        ax.axvspan(w_lo, w_hi, color="gray", alpha=0.15, zorder=0,
                  label="analysis window")

    ax = axes[0]
    _shade(ax)
    ax.plot(np.arange(len(data["j_ts"])), data["j_ts"], color="purple", lw=1.3)
    ax.axvline(j_idx, color="black", ls="--", lw=1.5, label="J candidate idx")
    ax.scatter([j_idx], [data["j_ts"][j_idx]], color="black", zorder=5, s=40)
    ax.set_ylabel("J coupling")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    _shade(ax)
    ax.plot(np.arange(len(data["accel_ts"])), data["accel_ts"], color="darkorange", lw=1.3)
    ax.axvline(j_idx, color="black", ls="--", lw=1.5)
    ax.scatter([accel_max_idx], [data["accel_ts"][accel_max_idx]],
              color="green", zorder=5, s=50, marker="^", label="accel max")
    ax.scatter([accel_min_idx], [data["accel_ts"][accel_min_idx]],
              color="red", zorder=5, s=50, marker="v", label="accel min")
    ax.set_ylabel("Acceleration")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    _shade(ax)
    ax.plot(np.arange(len(data["fr_ts"])), data["fr_ts"], color="steelblue", lw=1.3)
    ax.axvline(j_idx, color="black", ls="--", lw=1.5)
    if fr_peak_idx is not None:
        ax.scatter([fr_peak_idx], [data["fr_ts"][fr_peak_idx]],
                  color="green", zorder=5, s=50, marker="^", label="FR peak")
    if fr_trough_idx is not None:
        ax.scatter([fr_trough_idx], [data["fr_ts"][fr_trough_idx]],
                  color="red", zorder=5, s=50, marker="v", label="FR trough")
    ax.set_ylabel("Firing rate")
    ax.set_xlabel("Time bin (full recording; shaded = analysis window)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, n_max - 1)

    fig.suptitle(
        f"{session_id}  stim={stim}   |J-accel|={dist_accel}  |J-FR|={dist_fr}  "
        f"-> closer_to={closer_to}",
        fontweight="bold", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_dataset(data_folder, w_lo=350, w_hi=475, stim_min=0, stim_max_exclusive=3,
                  peak_threshold=1.75, smooth_sigma=5.0, sessions=None,
                  plots_dir=None):
    all_paths = find_file_recursive(data_folder, "per_reach_state.csv", verbose=False)
    whitelist = set(sessions) if sessions else None
    stim_range = range(stim_min, stim_max_exclusive)

    rows = []
    for path in all_paths:
        stim = _stim_from_path(path)
        if stim is None or stim not in stim_range:
            continue
        sid = _session_id_from_path(path)
        if whitelist and sid not in whitelist:
            continue

        data = load_session_pk_series(path, stim, w_lo, w_hi,
                                      peak_threshold, smooth_sigma)
        if data is None:
            continue

        j_idx, j_ratio, fr_peak_idx, fr_trough_idx = _raw_candidate_indices(
            data, peak_threshold, smooth_sigma)
        if j_idx is None or (fr_peak_idx is None and fr_trough_idx is None):
            continue

        # Raw max/min acceleration in the window -- two separate reference
        # points, same treatment as firing rate's peak/trough, rather than
        # collapsing to a single argmax(|accel|) "peak" index.
        w_lo, w_hi = data["w_lo"], data["w_hi"]
        accel_win = data["accel_ts"][w_lo:w_hi]
        accel_max_idx = w_lo + int(np.argmax(accel_win))
        accel_min_idx = w_lo + int(np.argmin(accel_win))

        dist_accel = min(abs(j_idx - accel_max_idx), abs(j_idx - accel_min_idx))
        fr_candidates = [d for d in (fr_peak_idx, fr_trough_idx) if d is not None]
        dist_fr = min(abs(j_idx - d) for d in fr_candidates)

        closer_to = ("accel" if dist_accel < dist_fr
                    else ("fr" if dist_fr < dist_accel else "tie"))

        rows.append({
            "session_id": sid, "stim": stim,
            "has_j_peak": bool(data["has_j_peak"]),
            "j_peak_ratio": float(j_ratio),
            "j_idx": int(j_idx),
            "accel_max_idx": accel_max_idx, "accel_min_idx": accel_min_idx,
            "fr_peak_idx": fr_peak_idx, "fr_trough_idx": fr_trough_idx,
            "dist_accel": int(dist_accel), "dist_fr": int(dist_fr),
            "closer_to": closer_to,
        })

        if plots_dir:
            subdir = {"accel": "accel_winning", "fr": "fr_winning", "tie": "tie"}[closer_to]
            save_path = os.path.join(plots_dir, subdir, f"stim{stim}_{sid}.png")
            _plot_verification(sid, stim, data, j_idx, accel_max_idx, accel_min_idx,
                               fr_peak_idx, fr_trough_idx, int(dist_accel), int(dist_fr),
                               closer_to, save_path)

    return pd.DataFrame(rows)


def _report_group(df: pd.DataFrame, label: str):
    n = len(df)
    print(f"\n{label}  (n={n})")
    print("-" * 60)
    if n == 0:
        print("  no sessions in this group")
        return

    da = df["dist_accel"].to_numpy(dtype=float)
    df_ = df["dist_fr"].to_numpy(dtype=float)
    n_accel = int((df["closer_to"] == "accel").sum())
    n_fr    = int((df["closer_to"] == "fr").sum())
    n_tie   = int((df["closer_to"] == "tie").sum())

    print(f"  median |J - nearest(accel min/max)| = {np.median(da):.1f} bins   "
         f"(mean {da.mean():.1f} \u00b1 {da.std():.1f})")
    print(f"  median |J - nearest(FR min/max)|    = {np.median(df_):.1f} bins   "
         f"(mean {df_.mean():.1f} \u00b1 {df_.std():.1f})")
    print(f"  closer to accel: {n_accel}/{n}   closer to FR: {n_fr}/{n}   ties: {n_tie}")

    if n >= 1 and not np.allclose(da, df_):
        try:
            wstat, p = wilcoxon(da, df_)
            if p < 0.05 and np.median(da) < np.median(df_):
                verdict = "accel significantly closer (p<0.05)"
            elif p < 0.05 and np.median(df_) < np.median(da):
                verdict = "FR significantly closer (p<0.05)"
            else:
                verdict = "no significant difference"
            print(f"  Wilcoxon signed-rank (|J-accel| vs |J-FR|): W={wstat:.0f}  p={p:.4f}"
                 f"  -> {verdict}")
        except Exception as e:
            print(f"  Wilcoxon signed-rank: n/a ({e})")
    else:
        print("  Wilcoxon signed-rank: n/a (identical distances)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_folder", required=True)
    p.add_argument("--window", nargs=2, type=int, default=[350, 475], metavar=("LO", "HI"))
    p.add_argument("--stim_min", type=int, default=0)
    p.add_argument("--stim_max_exclusive", type=int, default=3)
    p.add_argument("--peak_threshold", type=float, default=1.75)
    p.add_argument("--smooth_sigma", type=float, default=5.0)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--csv_out", default=None,
                  help="Optional path to save the per-session distance table.")
    p.add_argument("--plots_dir", default=None,
                  help="Optional directory to save per-session verification plots "
                       "into accel_winning/ and fr_winning/ (and tie/) subfolders.")
    args = p.parse_args()

    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    df = build_dataset(data_folder, args.window[0], args.window[1],
                       args.stim_min, args.stim_max_exclusive,
                       args.peak_threshold, args.smooth_sigma, args.sessions,
                       plots_dir=args.plots_dir)

    if df.empty:
        print("No usable sessions found.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("J-PEAK TIMING vs. KINEMATICS/FIRING-RATE")
    print(f"Window=[{args.window[0]}, {args.window[1]}]  "
         f"threshold={args.peak_threshold}  smooth_sigma={args.smooth_sigma}")
    print(f"Total sessions: {len(df)}")
    print("=" * 60)

    _report_group(df[df["has_j_peak"]],  "J-peak sessions")
    _report_group(df[~df["has_j_peak"]], "No/weak-peak sessions")
    _report_group(df,                    "All sessions (regardless of peak status)")

    if args.csv_out:
        df.sort_values(["session_id", "stim"]).to_csv(args.csv_out, index=False)
        print(f"\nPer-session table \u2192 {args.csv_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
