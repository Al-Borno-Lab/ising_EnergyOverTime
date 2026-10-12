"""
test_collective_j_timing.py
----------------------------
Which signal's timing best indicates the acceleration peak: J-coupling,
H-field, firing rate, or Ising energy?

Session inclusion does NOT require the P(K) collective-behavior flag
(is_collective) -- only a significant, gated J-coupling peak (has_j_peak,
same threshold_ratio/smooth_sigma as pk_j_analysis.py) is required, so the
comparison isn't restricted to the subset of sessions where the Ising model
already fits better than the independent model. is_collective is still
recorded per session in the output CSV for reference/filtering.

All candidate indices are compared against accel_max_idx, the TRUE signed
maximum of acceleration in the window (argmax(accel), NOT argmax(|accel|) --
pk_j_analysis.py's own accel_peak_idx uses the absolute-value version, which
can land on a bigger-magnitude trough instead of the actual peak):
  - J coupling  : j_peak_idx    (gated; has_j_peak required for inclusion)
  - H field     : h_peak_idx    (gated on has_h_peak; NaN if no sig. H peak)
  - Firing rate : nearest of fr_peak_idx / fr_trough_idx (gated; NaN if
                  neither detected)
  - Energy      : nearest of raw energy_max_idx / energy_min_idx in the
                  window (NOT gated by significance -- energy has no
                  detector in pk_j_analysis.py, so this mirrors how accel's
                  own max/min are read directly off the signal)

For each session the indicator with the smallest distance to accel_max_idx
"wins" (ties among the smallest broken as "tie"); pairwise Wilcoxon
signed-rank tests are run on every pair of indicators that both have data
for at least one common session.

Usage
-----
  python test_collective_j_timing.py --data_folder outputs/energy_decomp_Jul_10_svg \\
      --plots_dir outputs/energy_decomp_Jul_10_svg/notes/pk_event_analysis/all_signals_vs_accel \\
      --csv_out outputs/energy_decomp_Jul_10_svg/notes/pk_event_analysis/all_signals_vs_accel/timing_vs_accel.csv
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
from pk_j_analysis import load_session_pk_series, _stim_from_path, _session_id_from_path

INDICATORS = ["j", "h", "fr", "energy"]
LABELS = {"j": "J coupling", "h": "H field", "fr": "Firing rate", "energy": "Energy"}
COLORS = {"j": "purple", "h": "firebrick", "fr": "steelblue", "energy": "darkgoldenrod"}

plt.rcParams.update({
    "axes.grid":        False,
    "font.size":        20,
    "axes.titlesize":   22,
    "axes.labelsize":   20,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  16,
    "figure.titlesize": 22,
})


def _plot_verification(session_id, stim, data, j_idx, h_idx, accel_max_idx,
                       fr_peak_idx, fr_trough_idx, energy_max_idx, energy_min_idx,
                       dists, winner, save_path):
    """
    5-panel figure (J, H, acceleration, firing rate, energy) over the FULL
    recording, analysis window shaded, accel_max_idx marked in every panel
    so each indicator's distance to it can be seen in context.
    """
    w_lo, w_hi = data["w_lo"], data["w_hi"]
    n_max = max(len(data["j_ts"]), len(data.get("h_ts", [])), len(data["accel_ts"]),
               len(data["fr_ts"]), len(data["energy_ts"]))

    fig, axes = plt.subplots(5, 1, figsize=(11, 12.5), sharex=True, layout="constrained")

    def _shade(ax):
        ax.axvspan(w_lo, w_hi, color="gray", alpha=0.15, zorder=0)
        ax.axvline(accel_max_idx, color="black", ls="--", lw=1.5)

    ax = axes[0]
    _shade(ax)
    ax.plot(np.arange(len(data["j_ts"])), data["j_ts"], color=COLORS["j"], lw=1.3)
    ax.scatter([j_idx], [data["j_ts"][j_idx]], color=COLORS["j"], zorder=5, s=50,
              marker="o", label=f"J peak  |d|={dists['j']:.0f}")
    ax.set_ylabel("J coupling")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    _shade(ax)
    h_ts = data.get("h_ts", np.array([]))
    if len(h_ts) > 0:
        ax.plot(np.arange(len(h_ts)), h_ts, color=COLORS["h"], lw=1.3)
        if h_idx is not None:
            ax.scatter([h_idx], [h_ts[h_idx]], color=COLORS["h"], zorder=5, s=50,
                      marker="o", label=f"H peak  |d|={dists['h']:.0f}")
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.02, 0.9, "no significant H peak", transform=ax.transAxes,
                   fontsize=9, va="top")
    ax.set_ylabel("H field")

    ax = axes[2]
    _shade(ax)
    ax.plot(np.arange(len(data["accel_ts"])), data["accel_ts"], color="darkorange", lw=1.3)
    ax.scatter([accel_max_idx], [data["accel_ts"][accel_max_idx]],
              color="black", zorder=5, s=50, marker="*", label="accel max (true max, not |a|)")
    ax.set_ylabel("Acceleration")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    _shade(ax)
    ax.plot(np.arange(len(data["fr_ts"])), data["fr_ts"], color=COLORS["fr"], lw=1.3)
    if fr_peak_idx is not None:
        ax.scatter([fr_peak_idx], [data["fr_ts"][fr_peak_idx]],
                  color="green", zorder=5, s=50, marker="^", label="FR peak")
    if fr_trough_idx is not None:
        ax.scatter([fr_trough_idx], [data["fr_ts"][fr_trough_idx]],
                  color="red", zorder=5, s=50, marker="v", label="FR trough")
    ax.set_ylabel("Firing rate")
    if fr_peak_idx is not None or fr_trough_idx is not None:
        dfr_label = f"|d|={dists['fr']:.0f}" if not np.isnan(dists["fr"]) else None
        ax.legend(loc="upper right", fontsize=9, title=dfr_label)

    ax = axes[4]
    _shade(ax)
    ax.plot(np.arange(len(data["energy_ts"])), data["energy_ts"], color=COLORS["energy"], lw=1.3)
    if energy_max_idx is not None:
        ax.scatter([energy_max_idx], [data["energy_ts"][energy_max_idx]],
                  color="green", zorder=5, s=50, marker="^", label="Energy max")
    if energy_min_idx is not None:
        ax.scatter([energy_min_idx], [data["energy_ts"][energy_min_idx]],
                  color="red", zorder=5, s=50, marker="v", label="Energy min")
    ax.set_ylabel("Energy")
    ax.set_xlabel("Time bin (full recording; shaded = analysis window)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, n_max - 1)

    dist_str = "  ".join(f"{LABELS[k]}={dists[k]:.0f}" if not np.isnan(dists[k]) else f"{LABELS[k]}=n/a"
                         for k in INDICATORS)
    fig.suptitle(f"{session_id}  stim={stim}   dist-to-accel-max: {dist_str}   -> closest={winner}",
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
    n_seen = 0
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
        n_seen += 1

        # Only requirement for inclusion: a real, gated J-coupling peak.
        # (No is_collective requirement -- that's recorded but not filtered on.)
        if not data["has_j_peak"] or data["j_peak_idx"] is None:
            continue

        win_lo, win_hi = data["w_lo"], data["w_hi"]
        accel_win = data["accel_ts"][win_lo:win_hi]
        accel_max_idx = win_lo + int(np.argmax(accel_win))

        j_idx = data["j_peak_idx"]
        dist_j = float(abs(j_idx - accel_max_idx))

        h_idx = data["h_peak_idx"] if data["has_h_peak"] else None
        dist_h = float(abs(h_idx - accel_max_idx)) if h_idx is not None else np.nan

        fr_peak_idx = data["fr_peak_idx"]
        fr_trough_idx = data["fr_trough_idx"]
        fr_candidates = [d for d in (fr_peak_idx, fr_trough_idx) if d is not None]
        dist_fr = (float(min(abs(d - accel_max_idx) for d in fr_candidates))
                   if fr_candidates else np.nan)

        energy_ts = data["energy_ts"]
        if len(energy_ts) >= win_hi and win_hi > win_lo:
            energy_win = energy_ts[win_lo:win_hi]
            energy_max_idx = win_lo + int(np.argmax(energy_win))
            energy_min_idx = win_lo + int(np.argmin(energy_win))
            dist_energy = float(min(abs(energy_max_idx - accel_max_idx),
                                    abs(energy_min_idx - accel_max_idx)))
        else:
            energy_max_idx = energy_min_idx = None
            dist_energy = np.nan

        dists = {"j": dist_j, "h": dist_h, "fr": dist_fr, "energy": dist_energy}
        valid = {k: v for k, v in dists.items() if not np.isnan(v)}
        if len(valid) >= 2:
            min_val = min(valid.values())
            winners = [k for k, v in valid.items() if v == min_val]
            winner = winners[0] if len(winners) == 1 else "tie"
        elif len(valid) == 1:
            winner = next(iter(valid))
        else:
            winner = "none"

        pk_mets = data["pk_mets"]
        rows.append({
            "session_id": sid, "stim": stim,
            "j_peak_idx": int(j_idx),
            "h_peak_idx": h_idx,
            "accel_max_idx": int(accel_max_idx),
            "fr_peak_idx": fr_peak_idx, "fr_trough_idx": fr_trough_idx,
            "energy_max_idx": energy_max_idx, "energy_min_idx": energy_min_idx,
            "dist_j": dist_j, "dist_h": dist_h,
            "dist_fr": dist_fr, "dist_energy": dist_energy,
            "closest_indicator": winner,
            "is_collective": bool(data["is_collective"]),
            "r_ising": pk_mets.get("r_ising", np.nan),
            "r_independent": pk_mets.get("r_independent", np.nan),
            "ising_indep_dist": pk_mets.get("ising_indep_dist", np.nan),
        })

        if plots_dir:
            subdir = f"{winner}_winning" if winner not in ("tie", "none") else winner
            save_path = os.path.join(plots_dir, subdir, f"stim{stim}_{sid}.png")
            _plot_verification(sid, stim, data, j_idx, h_idx, accel_max_idx,
                               fr_peak_idx, fr_trough_idx, energy_max_idx, energy_min_idx,
                               dists, winner, save_path)

    print(f"\n{n_seen} session/stim pairs loaded; "
         f"{len(rows)} had a significant J-coupling peak (has_j_peak).")
    return pd.DataFrame(rows)


def _raw_max_min(ts: np.ndarray, w_lo: int, w_hi: int):
    """argmax/argmin of *ts* inside [w_lo, w_hi) -- no smoothing, no
    significance testing, just the literal highest and lowest points."""
    if len(ts) < w_hi or w_hi <= w_lo:
        return None, None
    win = ts[w_lo:w_hi]
    return w_lo + int(np.argmax(win)), w_lo + int(np.argmin(win))


def _dist_nearest(max_idx, min_idx, ref_idx):
    cands = [c for c in (max_idx, min_idx) if c is not None]
    return float(min(abs(c - ref_idx) for c in cands)) if cands else np.nan


def build_dataset_raw_extrema(data_folder, w_lo=350, w_hi=475, stim_min=0,
                              stim_max_exclusive=3, peak_threshold=1.75,
                              smooth_sigma=5.0, sessions=None, plots_dir=None):
    """
    Same comparison as build_dataset(), but with NO peak-finder algorithm
    anywhere: every signal (J, H, FR, Energy) is reduced to the nearest of
    its raw argmax/argmin within the window, exactly like Energy already
    was in build_dataset(). This drops the has_j_peak entry requirement too,
    so all session/stim pairs are eligible (not just the ones with a
    significant, gated J peak) -- an apples-to-apples comparison at the
    cost of losing any notion of "is this actually a meaningful event".
    """
    all_paths = find_file_recursive(data_folder, "per_reach_state.csv", verbose=False)
    whitelist = set(sessions) if sessions else None
    stim_range = range(stim_min, stim_max_exclusive)

    rows = []
    n_seen = 0
    for path in all_paths:
        stim = _stim_from_path(path)
        if stim is None or stim not in stim_range:
            continue
        sid = _session_id_from_path(path)
        if whitelist and sid not in whitelist:
            continue

        # peak_threshold/smooth_sigma are only used by load_session_pk_series
        # to compute has_j_peak/has_h_peak/etc, which this function ignores --
        # kept so the window/series extraction stays identical to build_dataset.
        data = load_session_pk_series(path, stim, w_lo, w_hi,
                                      peak_threshold, smooth_sigma)
        if data is None:
            continue
        n_seen += 1

        win_lo, win_hi = data["w_lo"], data["w_hi"]
        accel_max_idx, _ = _raw_max_min(data["accel_ts"], win_lo, win_hi)
        if accel_max_idx is None:
            continue

        j_max_idx, j_min_idx = _raw_max_min(data["j_ts"], win_lo, win_hi)
        dist_j = _dist_nearest(j_max_idx, j_min_idx, accel_max_idx)

        h_ts = data.get("h_ts", np.array([]))
        h_max_idx, h_min_idx = _raw_max_min(h_ts, win_lo, win_hi) if len(h_ts) > 0 else (None, None)
        dist_h = _dist_nearest(h_max_idx, h_min_idx, accel_max_idx)

        fr_max_idx, fr_min_idx = _raw_max_min(data["fr_ts"], win_lo, win_hi)
        dist_fr = _dist_nearest(fr_max_idx, fr_min_idx, accel_max_idx)

        energy_max_idx, energy_min_idx = _raw_max_min(data["energy_ts"], win_lo, win_hi)
        dist_energy = _dist_nearest(energy_max_idx, energy_min_idx, accel_max_idx)

        dists = {"j": dist_j, "h": dist_h, "fr": dist_fr, "energy": dist_energy}
        valid = {k: v for k, v in dists.items() if not np.isnan(v)}
        if len(valid) >= 2:
            min_val = min(valid.values())
            winners = [k for k, v in valid.items() if v == min_val]
            winner = winners[0] if len(winners) == 1 else "tie"
        elif len(valid) == 1:
            winner = next(iter(valid))
        else:
            winner = "none"

        pk_mets = data["pk_mets"]
        rows.append({
            "session_id": sid, "stim": stim,
            "j_max_idx": j_max_idx, "j_min_idx": j_min_idx,
            "h_max_idx": h_max_idx, "h_min_idx": h_min_idx,
            "accel_max_idx": int(accel_max_idx),
            "fr_max_idx": fr_max_idx, "fr_min_idx": fr_min_idx,
            "energy_max_idx": energy_max_idx, "energy_min_idx": energy_min_idx,
            "dist_j": dist_j, "dist_h": dist_h,
            "dist_fr": dist_fr, "dist_energy": dist_energy,
            "closest_indicator": winner,
            "is_collective": bool(data["is_collective"]),
            "has_j_peak": bool(data["has_j_peak"]),
            "r_ising": pk_mets.get("r_ising", np.nan),
            "r_independent": pk_mets.get("r_independent", np.nan),
            "ising_indep_dist": pk_mets.get("ising_indep_dist", np.nan),
        })

        if plots_dir:
            subdir = f"{winner}_winning" if winner not in ("tie", "none") else winner
            save_path = os.path.join(plots_dir, subdir, f"stim{stim}_{sid}.png")
            _plot_verification_raw(sid, stim, data, accel_max_idx,
                                   {"j": (j_max_idx, j_min_idx), "h": (h_max_idx, h_min_idx),
                                    "fr": (fr_max_idx, fr_min_idx),
                                    "energy": (energy_max_idx, energy_min_idx)},
                                   dists, winner, save_path)

    print(f"\n{n_seen} session/stim pairs loaded; "
         f"{len(rows)} usable (argmax/argmin, no peak-finder, no has_j_peak gate).")
    return pd.DataFrame(rows)


def _plot_verification_raw(session_id, stim, data, accel_max_idx, idxs, dists, winner, save_path):
    """
    5-panel figure (J, H, acceleration, firing rate, energy), all four
    non-accel signals marked with their raw argmax (^) / argmin (v) in the
    window -- no peak-finder markers here, unlike _plot_verification.
    """
    w_lo, w_hi = data["w_lo"], data["w_hi"]
    n_max = max(len(data["j_ts"]), len(data.get("h_ts", [])), len(data["accel_ts"]),
               len(data["fr_ts"]), len(data["energy_ts"]))

    fig, axes = plt.subplots(5, 1, figsize=(11, 12.5), sharex=True, layout="constrained")

    def _shade(ax):
        ax.axvspan(w_lo, w_hi, color="gray", alpha=0.15, zorder=0)
        ax.axvline(accel_max_idx, color="black", ls="--", lw=1.5)

    def _panel(ax, ts, key, ylabel):
        _shade(ax)
        ax.plot(np.arange(len(ts)), ts, color=COLORS[key], lw=1.3)
        max_idx, min_idx = idxs[key]
        if max_idx is not None:
            ax.scatter([max_idx], [ts[max_idx]], color="green", zorder=5, s=50, marker="^")
        if min_idx is not None:
            ax.scatter([min_idx], [ts[min_idx]], color="red", zorder=5, s=50, marker="v")
        d = dists[key]
        ax.set_ylabel(ylabel + (f"\n|d|={d:.0f}" if not np.isnan(d) else "\n|d|=n/a"))

    _panel(axes[0], data["j_ts"], "j", "J coupling")
    h_ts = data.get("h_ts", np.array([]))
    if len(h_ts) > 0:
        _panel(axes[1], h_ts, "h", "H field")
    else:
        _shade(axes[1]); axes[1].set_ylabel("H field")

    ax = axes[2]
    _shade(ax)
    ax.plot(np.arange(len(data["accel_ts"])), data["accel_ts"], color="darkorange", lw=1.3)
    ax.scatter([accel_max_idx], [data["accel_ts"][accel_max_idx]],
              color="black", zorder=5, s=50, marker="*")
    ax.set_ylabel("Acceleration\n(ref: true max)")

    _panel(axes[3], data["fr_ts"], "fr", "Firing rate")
    _panel(axes[4], data["energy_ts"], "energy", "Energy")
    axes[4].set_xlabel("Time bin (full recording; shaded = analysis window)")
    axes[4].set_xlim(0, n_max - 1)

    dist_str = "  ".join(f"{LABELS[k]}={dists[k]:.0f}" if not np.isnan(dists[k]) else f"{LABELS[k]}=n/a"
                         for k in INDICATORS)
    fig.suptitle(f"{session_id}  stim={stim}  [argmax/argmin only]\n"
                f"dist-to-accel-max: {dist_str}   -> closest={winner}",
                fontweight="bold", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_boxwhisker(df: pd.DataFrame, subtitle: str, save_path: str):
    """
    Single box-and-whisker figure comparing the four indicators' distance
    (in bins) to the true acceleration max, side by side. Points are
    individual sessions (jittered), boxes show the distribution.
    """
    data = [df[f"dist_{k}"].dropna().to_numpy(dtype=float) for k in INDICATORS]
    colors = [COLORS[k] for k in INDICATORS]

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2),
                    widths=0.55, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(data, colors), 1):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="gray",
                  alpha=0.6, s=30, zorder=3)

    ax.set_xticks(range(1, len(INDICATORS) + 1))
    ax.set_xticklabels([f"{LABELS[k]}\n(n={len(data[i])})"
                        for i, k in enumerate(INDICATORS)])
    ax.set_ylabel("Distance to true accel max (bins)")
    fig.suptitle(f"Timing distance to acceleration peak\n{subtitle}", fontweight="bold")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    svg_path = os.path.splitext(save_path)[0] + ".svg"
    with plt.rc_context({"svg.fonttype": "path"}):
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _report(df: pd.DataFrame, title: str = None):
    n = len(df)
    print("=" * 70)
    print(title or f"J-PEAK SESSIONS (no collective-behavior gate): "
                   f"WHICH SIGNAL BEST INDICATES TRUE ACCEL MAX?  (n={n})")
    print("=" * 70)
    if n == 0:
        print("  no sessions available")
        return

    for k in INDICATORS:
        vals = df[f"dist_{k}"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            print(f"  {LABELS[k]:12s}: no sessions with a usable event")
            continue
        print(f"  {LABELS[k]:12s}: n={len(vals):2d}  median={np.median(vals):5.1f} bins  "
             f"mean={vals.mean():5.1f}\u00b1{vals.std():.1f}")

    print()
    counts = df["closest_indicator"].value_counts()
    print("  closest-indicator counts (per session, among available indicators):")
    for k in INDICATORS + ["tie", "none"]:
        if k in counts.index:
            print(f"    {LABELS.get(k, k):12s}: {counts[k]}")

    print()
    print("  pairwise Wilcoxon signed-rank tests (sessions with both indicators available):")
    for i in range(len(INDICATORS)):
        for j in range(i + 1, len(INDICATORS)):
            a, b = INDICATORS[i], INDICATORS[j]
            sub = df[[f"dist_{a}", f"dist_{b}"]].dropna()
            label = f"{LABELS[a]} vs {LABELS[b]}"
            if len(sub) < 1:
                print(f"    {label}: n/a (no sessions with both)")
                continue
            da = sub[f"dist_{a}"].to_numpy(dtype=float)
            db = sub[f"dist_{b}"].to_numpy(dtype=float)
            if np.allclose(da, db):
                print(f"    {label} (n={len(sub)}): n/a (identical distances)")
                continue
            try:
                wstat, p = wilcoxon(da, db)
                if p < 0.05:
                    verdict = f"{LABELS[a]} closer" if np.median(da) < np.median(db) else f"{LABELS[b]} closer"
                else:
                    verdict = "no significant difference"
                print(f"    {label} (n={len(sub)}): W={wstat:.0f}  p={p:.4f}  -> {verdict}")
            except Exception as e:
                print(f"    {label}: n/a ({e})")


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
                       "into j_winning/, h_winning/, fr_winning/, energy_winning/ "
                       "(and tie/, none/) subfolders.")
    p.add_argument("--boxplot_dir", default=None,
                  help="Optional directory to save the two summary box-and-whisker "
                       "figures (all sessions, and collective-behavior-only sessions). "
                       "Defaults to the csv_out directory, then plots_dir, if not given.")
    p.add_argument("--skip_j_gated", action="store_true",
                  help="Skip the secondary has_j_peak-gated comparison entirely and "
                       "only run the unbiased (argmax/argmin, no peak-finder gate) "
                       "primary comparison.")
    p.add_argument("--raw_csv_out", default=None)
    p.add_argument("--raw_plots_dir", default=None)
    p.add_argument("--raw_boxplot_dir", default=None)
    args = p.parse_args()

    # FIX (verification audit, §1.5): build_dataset()'s has_j_peak gate only
    # includes sessions where a J-coupling peak was already detected, while
    # competing indicators (H, firing rate, energy) are gated/NaN'd
    # independently elsewhere -- this biases the "closest indicator" tally
    # and pairwise comparisons toward J. build_dataset_raw_extrema() (no
    # peak-finder gate, plain argmax/argmin) has no such bias and is now the
    # PRIMARY/headline comparison; the has_j_peak-gated version is kept only
    # as an explicitly-labeled secondary "conditional on J showing a peak"
    # analysis, opt-out via --skip_j_gated.
    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    df = build_dataset_raw_extrema(data_folder, args.window[0], args.window[1],
                       args.stim_min, args.stim_max_exclusive,
                       args.peak_threshold, args.smooth_sigma, args.sessions,
                       plots_dir=args.plots_dir)

    if df.empty:
        print("No usable sessions found.", file=sys.stderr)
        return 1

    print(f"Window=[{args.window[0]}, {args.window[1]}]  "
         f"threshold={args.peak_threshold}  smooth_sigma={args.smooth_sigma}")
    _report(df, title=f"PRIMARY (unbiased): argmax/argmin only, all sessions "
                      f"(no peak-finder, no has_j_peak gate)  (n={len(df)})")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        df.sort_values(["session_id", "stim"]).to_csv(args.csv_out, index=False)
        print(f"\nPer-session table \u2192 {args.csv_out}")

    boxplot_dir = (args.boxplot_dir
                   or (os.path.dirname(args.csv_out) if args.csv_out else None)
                   or args.plots_dir)
    if boxplot_dir:
        all_path = os.path.join(boxplot_dir, "boxwhisker_all_sessions.png")
        _plot_boxwhisker(df, f"All sessions, unbiased comparison (n={len(df)})", all_path)
        print(f"Box-and-whisker (all sessions, primary/unbiased) \u2192 {all_path}")

        df_collective = df[df["is_collective"]]
        collective_path = os.path.join(boxplot_dir, "boxwhisker_collective_only.png")
        _plot_boxwhisker(df_collective,
                         f"Collective-behavior sessions only, unbiased comparison (n={len(df_collective)})",
                         collective_path)
        print(f"Box-and-whisker (collective only, primary/unbiased) \u2192 {collective_path}")

    if not args.skip_j_gated:
        print()
        df_jgated = build_dataset(
            data_folder, args.window[0], args.window[1],
            args.stim_min, args.stim_max_exclusive,
            args.peak_threshold, args.smooth_sigma, args.sessions,
            plots_dir=args.raw_plots_dir)

        if df_jgated.empty:
            print("No sessions with a significant J peak found "
                  "(secondary, J-gated comparison skipped).", file=sys.stderr)
        else:
            _report(df_jgated, title=f"SECONDARY (conditional on J showing a peak, "
                                     f"selection-biased toward J -- see verification "
                                     f"audit \u00a71.5): has_j_peak-gated sessions only "
                                     f"(n={len(df_jgated)})")

            if args.raw_csv_out:
                os.makedirs(os.path.dirname(args.raw_csv_out), exist_ok=True)
                df_jgated.sort_values(["session_id", "stim"]).to_csv(args.raw_csv_out, index=False)
                print(f"\nJ-gated (secondary) per-session table \u2192 {args.raw_csv_out}")

            jgated_boxplot_dir = (args.raw_boxplot_dir
                              or (os.path.dirname(args.raw_csv_out) if args.raw_csv_out else None)
                              or args.raw_plots_dir
                              or boxplot_dir)
            if jgated_boxplot_dir:
                all_path = os.path.join(jgated_boxplot_dir, "boxwhisker_all_sessions_jgated_SECONDARY.png")
                _plot_boxwhisker(df_jgated, f"SECONDARY, J-gated (biased toward J), all sessions (n={len(df_jgated)})",
                                 all_path)
                print(f"Box-and-whisker (J-gated, secondary, all sessions) \u2192 {all_path}")

                df_jgated_collective = df_jgated[df_jgated["is_collective"]]
                collective_path = os.path.join(jgated_boxplot_dir,
                                               "boxwhisker_collective_only_jgated_SECONDARY.png")
                _plot_boxwhisker(df_jgated_collective,
                                 f"SECONDARY, J-gated (biased toward J), collective-behavior sessions "
                                 f"(n={len(df_jgated_collective)})",
                                 collective_path)
                print(f"Box-and-whisker (J-gated, secondary, collective only) \u2192 {collective_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
