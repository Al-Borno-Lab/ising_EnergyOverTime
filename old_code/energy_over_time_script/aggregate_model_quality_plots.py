#!/usr/bin/env python3
"""
Aggregate model quality summary plots across multiple sessions.

Scans a root directory tree for session sub-folders that contain the CSVs
written by visualization.plot_model_quality_summary, groups them by their
folder suffix (reach phase / stim label, e.g. begin_reach, full_reach), and
produces one integrated 4-panel figure per group.

Expected directory layout produced by process_matlab_file_manyTimes_sbatch.sh:

    <root>/
      <experiment>_rep1/
        begin_reach/   <- model_quality_summary_pairwise.csv, J_values.csv, …
        mid_reach/
        full_reach/
      <experiment>_rep2/
        begin_reach/
        …

Each leaf folder with model quality CSVs becomes one session. Folders sharing
the same last component (e.g. "full_reach") form one stim group → one figure.

Usage:
    python aggregate_model_quality_plots.py <root_dir> [--output_dir OUT] [--max_k 20]
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REQUIRED_CSV = "model_quality_summary_pairwise.csv"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_session_dirs(root: Path) -> list:
    """Walk root and return every directory that contains the model quality CSVs."""
    found = []
    for dirpath, _, files in os.walk(root):
        if REQUIRED_CSV in files:
            found.append(Path(dirpath))
    return sorted(found)


def group_by_stim(session_dirs: list) -> dict:
    """Group dirs by their last path component (reach-phase / stim label)."""
    groups = defaultdict(list)
    for d in session_dirs:
        groups[d.name].append(d)
    return dict(groups)


def load_csv(session_dir: Path, name: str):
    p = session_dir / name
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def session_label(sdir: Path) -> str:
    """Human-readable label: parent folder name (rep) + leaf."""
    return sdir.parent.name


def tab_colors(n: int) -> list:
    cmap = plt.cm.get_cmap("tab10", max(n, 1))
    return [cmap(i % 10) for i in range(n)]


# ---------------------------------------------------------------------------
# Main plotting function (one figure per stim group)
# ---------------------------------------------------------------------------

def plot_stim_summary(stim_label: str, session_dirs: list, output_dir: Path,
                      max_k: int = None):
    """
    4-panel integrated figure for one stim group:
      (a) Pairwise k=2 correlations – one scatter per session
      (b) J coupling distribution – overlaid histograms + pooled outline
      (c) Triplet k=3 correlations – one scatter per session
      (d) P(K) – thin session lines + bold mean for data / Ising / independent
    """
    n = len(session_dirs)
    colors = tab_colors(n)
    labels = [session_label(d) for d in session_dirs]

    # ---- load per-session data ----
    pair_orig_list, pair_model_list = [], []
    trip_orig_list, trip_model_list = [], []
    J_list = []
    pk_list = []   # each entry: (label, color, K, P_data, P_ising, P_indep)

    for sdir, color, lab in zip(session_dirs, colors, labels):
        df = load_csv(sdir, "model_quality_summary_pairwise.csv")
        if df is not None:
            pair_orig_list.append((lab, color, df["pairwise_corr_orig"].values,
                                    df["pairwise_corr_model"].values))

        df = load_csv(sdir, "model_quality_summary_triplet.csv")
        if df is not None:
            trip_orig_list.append((lab, color, df["triplet_corr_orig"].values,
                                    df["triplet_corr_model"].values))

        df = load_csv(sdir, "model_quality_summary_J_values.csv")
        if df is not None:
            J_list.append((lab, color, df["J"].values))

        df = load_csv(sdir, "model_quality_summary_P_K.csv")
        if df is not None:
            # handle old column name P_independent_MF
            indep_col = "P_independent" if "P_independent" in df.columns else "P_independent_MF"
            pk_list.append((lab, color,
                             df["K"].values,
                             df["P_data"].values,
                             df["P_ising"].values,
                             df[indep_col].values))

    # ---- layout ----
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.15, 1.0, 1.15],
                           width_ratios=[1, 1],
                           hspace=0.42, wspace=0.32)

    # ---- (a) Pairwise ----
    ax_a = fig.add_subplot(gs[0, :])
    all_vals = []
    for lab, color, orig, mod in pair_orig_list:
        ax_a.scatter(orig, mod, s=8, alpha=0.35, color=color,
                     edgecolors="none", label=lab)
        all_vals.extend(orig.tolist())
        all_vals.extend(mod.tolist())

    if all_vals:
        lo = min(min(all_vals), 0.0)
        hi = max(max(all_vals), 0.0)
        ax_a.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="identity", zorder=5)

    ax_a.set_xlabel(r"measured $C_{ij}$")
    ax_a.set_ylabel(r"reconstructed $C_{ij}$")
    ax_a.set_title(f"[{stim_label}]  Pairwise correlation ($k=2$)  —  {n} sessions")
    ax_a.grid(alpha=0.3)
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")
    if n <= 12:
        ax_a.legend(loc="upper left", fontsize=7, ncol=max(1, n // 6))

    # ---- (b) J distribution ----
    ax_b = fig.add_subplot(gs[1, :])
    for lab, color, J in J_list:
        n_bins = min(50, max(10, len(J) // 5))
        ax_b.hist(J, bins=n_bins, density=True, color=color,
                  alpha=0.35, edgecolor="none", label=lab)

    if J_list:
        all_J = np.concatenate([j for _, _, j in J_list])
        ax_b.hist(all_J, bins=min(60, max(15, len(all_J) // 20)),
                  density=True, histtype="step", color="black",
                  lw=2.0, label="pooled", zorder=5)

    ax_b.set_xlabel(r"coupling $J$")
    ax_b.set_ylabel(r"$P(J)$")
    ax_b.set_title(f"[{stim_label}]  Distribution of pairwise couplings $J$  —  {n} sessions")
    ax_b.grid(alpha=0.3)
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")
    if n <= 12:
        ax_b.legend(loc="upper right", fontsize=7, ncol=max(1, n // 6))

    # ---- (c) Triplet ----
    ax_c = fig.add_subplot(gs[2, 0])
    all_trip = []
    for lab, color, orig, mod in trip_orig_list:
        ax_c.scatter(orig, mod, s=8, alpha=0.35, color=color, edgecolors="none")
        all_trip.extend(orig.tolist())
        all_trip.extend(mod.tolist())

    if all_trip:
        lo3, hi3 = min(all_trip), max(all_trip)
        ax_c.plot([lo3, hi3], [lo3, hi3], "k--", lw=1.2)

    ax_c.set_xlabel(r"measured $\langle \sigma_i \sigma_j \sigma_k \rangle$")
    ax_c.set_ylabel(r"predicted $\langle \sigma_i \sigma_j \sigma_k \rangle$")
    ax_c.set_title(f"[{stim_label}]  Triplet correlation ($k=3$)")
    ax_c.grid(alpha=0.3)
    ax_c.text(0.02, 0.98, "(c)", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

    # ---- (d) P(K) ----
    ax_d = fig.add_subplot(gs[2, 1])
    eps = 1e-12

    # Align K axes (pad shorter arrays)
    if pk_list:
        K_max = max(entry[2][-1] for entry in pk_list)
        k_ref = np.arange(K_max + 1)

    data_stack, ising_stack, indep_stack = [], [], []

    for lab, color, K, P_data, P_ising, P_indep in pk_list:
        # Pad to common length
        def _pad(arr, target):
            out = np.zeros(target + 1)
            out[:len(arr)] = arr
            return out

        pd_ = _pad(P_data, K_max)
        pi_ = _pad(P_ising, K_max)
        pin_ = _pad(P_indep, K_max)

        # Thin per-session lines
        ax_d.semilogy(k_ref, np.maximum(pd_, eps), "o-",
                      color=color, ms=2, lw=0.7, alpha=0.45)
        ax_d.semilogy(k_ref, np.maximum(pi_, eps), "s--",
                      color=color, ms=2, lw=0.7, alpha=0.45)
        ax_d.semilogy(k_ref, np.maximum(pin_, eps), "-",
                      color=color, ms=2, lw=0.7, alpha=0.25)

        data_stack.append(pd_)
        ising_stack.append(pi_)
        indep_stack.append(pin_)

    # Bold mean lines
    if data_stack:
        ax_d.semilogy(k_ref, np.maximum(np.mean(data_stack, axis=0), eps),
                      "o-", color="blue", ms=5, lw=2.0, label="data (mean)")
        ax_d.semilogy(k_ref, np.maximum(np.mean(ising_stack, axis=0), eps),
                      "o-", color="red", ms=5, lw=2.0, label="Ising (mean)")
        ax_d.semilogy(k_ref, np.maximum(np.mean(indep_stack, axis=0), eps),
                      "-", color="black", lw=2.0, label="independent (mean)")

    k_hi = K_max if max_k is None else min(K_max, max_k)
    ax_d.set_xlim(-0.5, k_hi + 0.5)
    ax_d.set_xlabel(r"$K$ (simultaneous spikes per bin)")
    ax_d.set_ylabel(r"$P(K)$")
    ax_d.set_title(f"[{stim_label}]  $P(K)$ — data / Ising / independent")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(alpha=0.3)
    ax_d.text(0.02, 0.98, "(d)", transform=ax_d.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

    # ---- finish ----
    fig.suptitle(
        f"Aggregate Model Quality — {stim_label} — {n} sessions",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    out_path = output_dir / f"aggregate_model_quality_{stim_label}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ---- pooled CSVs ----
    if pair_orig_list:
        pd.DataFrame({
            "session": sum([[lab] * len(orig) for lab, _, orig, _ in pair_orig_list], []),
            "stim_group": stim_label,
            "pairwise_corr_orig": np.concatenate([orig for _, _, orig, _ in pair_orig_list]),
            "pairwise_corr_model": np.concatenate([mod for _, _, _, mod in pair_orig_list]),
        }).to_csv(output_dir / f"aggregate_pairwise_{stim_label}.csv", index=False)

    if trip_orig_list:
        pd.DataFrame({
            "session": sum([[lab] * len(orig) for lab, _, orig, _ in trip_orig_list], []),
            "stim_group": stim_label,
            "triplet_corr_orig": np.concatenate([orig for _, _, orig, _ in trip_orig_list]),
            "triplet_corr_model": np.concatenate([mod for _, _, _, mod in trip_orig_list]),
        }).to_csv(output_dir / f"aggregate_triplet_{stim_label}.csv", index=False)

    if J_list:
        pd.DataFrame({
            "session": sum([[lab] * len(J) for lab, _, J in J_list], []),
            "stim_group": stim_label,
            "J": np.concatenate([J for _, _, J in J_list]),
        }).to_csv(output_dir / f"aggregate_J_{stim_label}.csv", index=False)

    if data_stack:
        pd.DataFrame({
            "K": k_ref,
            "P_data_mean": np.mean(data_stack, axis=0),
            "P_data_std": np.std(data_stack, axis=0),
            "P_ising_mean": np.mean(ising_stack, axis=0),
            "P_ising_std": np.std(ising_stack, axis=0),
            "P_independent_mean": np.mean(indep_stack, axis=0),
            "P_independent_std": np.std(indep_stack, axis=0),
            "n_sessions": len(data_stack),
        }).to_csv(output_dir / f"aggregate_P_K_{stim_label}.csv", index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_dir",
                        help="Root folder containing session sub-folders with model quality CSVs")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save aggregate figures (default: <root_dir>/aggregate_plots)")
    parser.add_argument("--max_k", type=int, default=None,
                        help="Cap P(K) x-axis at this value (e.g. 20 for large N). Default: full range.")
    args = parser.parse_args()

    root = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root / "aggregate_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = find_session_dirs(root)
    if not session_dirs:
        print(f"No session directories containing '{REQUIRED_CSV}' found under:\n  {root}")
        return

    groups = group_by_stim(session_dirs)
    print(f"Found {len(session_dirs)} session folders across {len(groups)} stim groups:")
    for label, dirs in sorted(groups.items()):
        print(f"  [{label}]  {len(dirs)} sessions")

    for stim_label, dirs in sorted(groups.items()):
        print(f"\nPlotting [{stim_label}] ...")
        plot_stim_summary(stim_label, dirs, output_dir, max_k=args.max_k)

    print(f"\nAll aggregate plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
