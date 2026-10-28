"""PI summary figure from the merged E12 tables (pre-registered, exclusions applied, n = 21 sessions).

    python docs/pi_figures/make_fig_field_dip.py 12_field_vs_rate_indicator/assets/tables docs/pi_figures/field_dip_E12
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIG = [("FT", "Field term\n(Ising, coupled)", "#1f4e99"), ("FT0", "Field term\n(no coupling)", "#e0861f"),
       ("POLb", "Polarization\nindex (no model)", "#2a9d8f"), ("FR", "Average\nfiring rate", "#c0392b")]


def main(tdir, out):
    m = pd.read_csv(f"{tdir}/primary_session_metrics.csv")
    fam = pd.read_csv(f"{tdir}/primary_primary_family.csv")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rng = np.random.default_rng(0)

    a = ax[0]
    for i, (s, lab, c) in enumerate(SIG):
        d = m[m.signal == s]
        a.bar(i - .18, d.hit.mean() * 100, .34, color=c)
        a.bar(i + .18, d.hit_c.mean() * 100, .34, color="#c9ced3")
        a.text(i - .18, d.hit.mean() * 100 + 2, f"{d.hit.mean()*100:.0f}%", ha="center", fontsize=8)
        a.text(i + .18, d.hit_c.mean() * 100 + 2, f"{d.hit_c.mean()*100:.0f}%", ha="center", fontsize=8, color="#666")
    a.set_xticks(range(len(SIG)), [l for _, l, _ in SIG])
    a.set_ylim(0, 85)
    a.set_ylabel("% of low points in the 200 ms before peak velocity")
    a.set_title("a  Is the low point locked to the speeding-up phase?\n(colour = reach, grey = same search at rest)", loc="left", fontsize=10)

    b = ax[1]
    for i, (s, lab, c) in enumerate(SIG):
        d = m[m.signal == s].dev_ms
        b.scatter(i + rng.uniform(-.12, .12, len(d)), d, s=14, color=c, alpha=.8)
        b.plot([i - .25, i + .25], [d.median()] * 2, color="k", lw=2)
        b.text(i + .28, d.median(), f"{d.median():.0f} ms", va="center", fontsize=8)
    dec = m[m.signal == "DEC"].dev_ms.median()
    b.axhline(dec, color="#888", ls=":", lw=1)
    b.text(-0.45, dec - 14, f"supervised decoder (ceiling): {dec:.0f} ms", fontsize=7.5, color="#666", ha="left", va="top")
    b.set_xticks(range(len(SIG)), [l for _, l, _ in SIG])
    b.set_ylabel("distance of low point from peak acceleration (ms)\nper session")
    b.set_title("b  How close to peak acceleration?", loc="left", fontsize=10)

    def p(test, A, B):
        r = fam[(fam.test == test) & (fam.A == A) & (fam.B == B)]
        return float(r.p_bh.iloc[0]) if len(r) else float("nan")
    b.annotate("", xy=(0, 640), xytext=(3, 640), arrowprops=dict(arrowstyle="-", color="k"))
    b.text(1.5, 650, f"field vs average rate: p(FDR) = {p('T2','FT','FR'):.3f}", ha="center", fontsize=8)
    b.text(1, 600, "field vs no-coupling & polarization: no detectable difference", ha="center", fontsize=7.5, color="#444")
    b.set_ylim(-20, 700)

    fig.text(0.01, -0.06,
             "E12, pre-registered; 21 sessions x 3 blocks, 10-ms bins, excluded reaches dropped; session-level Wilcoxon, BH-FDR over 12 tests. "
             "Field term = -sum_i h_i(t) s_i from the time-varying-field model.\nThe dip is a polarization effect (cells moving away from 50/50 in both "
             "directions): the no-coupling field term and a model-free polarization index time it as well as the Ising field term.",
             fontsize=7.5, color="#555", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
