"""E12 preliminary figure: field-term trough vs population firing-rate trough as a marker of peak acceleration.

Reads the E09 per-condition trough table (63 conditions = 21 sessions x 3 blocks) and draws a 3-panel summary.
Session-level statistics (conditions averaged within session, n = 21), two-sided Wilcoxon signed-rank.
Exploratory: not pre-registered, no multiple-comparison correction.

    python make_pi_figure.py <field_term_trough_timing.csv> <out_prefix>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FIELD, RATE, CTRL = "#d9822b", "#4a4f55", "#b8bec4"
BIN_MS = 10


def main(csv, out):
    d = pd.read_csv(csv)
    s = d.groupby("session").mean(numeric_only=True)
    n = len(s)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # A: trough timing per condition
    a = ax[0]
    edges = np.arange(-500, 1010, 25)
    a.axvspan(-200, 0, color="#e8eef7", zorder=0)
    a.hist(d.h_t * BIN_MS, bins=edges, color=FIELD, alpha=.85, label="field term")
    a.hist(d.fr_t * BIN_MS, bins=edges, histtype="step", lw=1.8, color=RATE,
           label="population firing rate")
    a.axvline(0, color="k", lw=1)
    a.axvline(-40, color="#3b6fb6", lw=1, ls=":")
    a.text(-195, a.get_ylim()[1] * .97, "200 ms before\npeak velocity", va="top", fontsize=8, color="#3b6fb6")
    a.text(-30, a.get_ylim()[1] * .45, "median peak\nacceleration", fontsize=7.5, color="#3b6fb6")
    a.set_xlabel("time of trough relative to peak velocity (ms)")
    a.set_ylabel("conditions (of 63)")
    a.set_title("a  When does each signal reach its minimum?", loc="left", fontweight="bold")
    a.legend(frameon=False, loc="upper right", fontsize=8)

    # B: fraction in pre-peak window, real vs control windows at rest
    b = ax[1]
    vals = [d.h_pre.mean(), d.h_pre_ctrl.mean(), d.fr_pre.mean(), d.fr_pre_ctrl.mean()]
    xs = [0, 0.8, 2.2, 3.0]
    b.bar(xs, [v * 100 for v in vals], width=.7, color=[FIELD, CTRL, RATE, CTRL],
          edgecolor=[FIELD, FIELD, RATE, RATE], linewidth=1.2)
    for x, v in zip(xs, vals):
        b.text(x, v * 100 + 1.5, f"{v*100:.0f}%", ha="center", fontsize=8.5)
    p_h = wilcoxon(s.h_pre, s.h_pre_ctrl).pvalue
    p_r = wilcoxon(s.fr_pre, s.fr_pre_ctrl).pvalue
    b.text(0.4, 75, f"p = {p_h:.1g}", ha="center", fontsize=8)
    b.text(2.6, 75, f"p = {p_r:.2f} (n.s.)", ha="center", fontsize=8)
    b.set_xticks(xs, ["reach", "rest\ncontrol", "reach", "rest\ncontrol"])
    b.text(0.4, -17, "field term", ha="center", color=FIELD, fontweight="bold", transform=b.transData, clip_on=False)
    b.text(2.6, -17, "firing rate", ha="center", color=RATE, fontweight="bold", transform=b.transData, clip_on=False)
    b.set_ylim(0, 85)
    b.set_ylabel("% troughs in 200 ms before peak velocity")
    b.set_title("b  Is the trough locked to acceleration?", loc="left", fontweight="bold")

    # C: depth relative to each signal's own rest control, per session
    c = ax[2]
    rh, rf = s.h_depth / s.h_depth_ctrl, s.fr_depth / s.fr_depth_ctrl
    for yh, yf in zip(rh, rf):
        c.plot([0, 1], [yh, yf], color="#c9cdd2", lw=.8, zorder=1)
    c.scatter(np.zeros(n), rh, color=FIELD, s=22, zorder=2)
    c.scatter(np.ones(n), rf, color=RATE, s=22, zorder=2)
    c.plot([-.15, .15], [rh.median()] * 2, color=FIELD, lw=2.5)
    c.plot([.85, 1.15], [rf.median()] * 2, color=RATE, lw=2.5)
    c.axhline(1, color="k", lw=.7, ls="--")
    c.set_yscale("log")
    c.set_xticks([0, 1], ["field term", "firing rate"])
    c.set_xlim(-.5, 1.5)
    p_c = wilcoxon(rh, rf).pvalue
    c.set_ylim(top=c.get_ylim()[1] * 8)
    c.text(.5, c.get_ylim()[1] * .7, f"median {rh.median():.1f}x vs {rf.median():.1f}x\n"
           f"field higher in {(rh > rf).sum()}/{n}, p = {p_c:.2f}", ha="center", va="top", fontsize=8)
    c.set_ylabel("trough depth / depth at rest (per session)")
    c.set_title("c  How deep, relative to its own noise?", loc="left", fontweight="bold")

    fig.text(0.01, -0.04,
             "21 sessions x 3 blocks = 63 conditions, 10-ms bins, exclusions applied. Field term = time-varying-field part of the "
             "Ising energy, -sum_i h_i(t) s_i; firing rate = population-average spike probability. Rest controls = the same "
             "search in windows shifted -1.5/-2/-2.5 s.\nStatistics are session-level (n = 21) two-sided Wilcoxon tests. "
             "EXPLORATORY: computed post hoc from the E09 table, not pre-registered, no FDR correction. Formal test planned as E12.",
             fontsize=7.5, color="#555", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"field pre {vals[0]:.2f} vs ctrl {vals[1]:.2f} p={p_h:.2g}; rate pre {vals[2]:.2f} vs ctrl {vals[3]:.2f} p={p_r:.2g}; "
          f"depth ratio {rh.median():.2f} vs {rf.median():.2f} p={p_c:.2g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
