"""S6: exact held-out surprise of M1 (time-varying fields + J) vs M0 (time-varying fields, J = 0), per reach, from the
cluster per_reach_surprise_<S>.csv files (each reach scored by models fitted without it).
    python m0_vs_m1.py DATA_DIR OUT_DIR"""
import sys, os, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, OUT = sys.argv[1], os.path.join(sys.argv[2], "S6_model_comparison_M0_vs_M1"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "entropy/per_reach_surprise_*.csv")))])
rows, G = [], {}
for w in ("trial", "rest", "reach"):
    P[f"gain_{w}"] = P[f"sur_M0_{w}"] - P[f"sur_M1_{w}"]
    c = P.groupby("key").agg(g=(f"gain_{w}", "mean"), m0=(f"sur_M0_{w}", "mean")); s = P.groupby("session")[f"gain_{w}"].mean(); G[w] = s
    rows.append(dict(window=w, sessions_M1_better=f"{int((s > 0).sum())}/{len(s)}", conditions_M1_better=f"{int((c.g > 0).sum())}/{len(c)}",
                     median_gain_bits_per_neuron_per_bin=round(float(s.median()), 4), relative_gain=f"{100 * float((c.g / c.m0).median()):.1f}%",
                     p_wilcoxon_sessions=float(wilcoxon(s).pvalue)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "M1_vs_M0_exact.csv"), index=False); print(T.to_string(index=False))
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for i, w in enumerate(("rest", "reach", "trial")):
    v = G[w].values; ax.scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=16, color="#1f4e99", alpha=.75)
    ax.plot([i - .25, i + .25], [np.median(v)] * 2, color="k", lw=2)
ax.axhline(0, color="#999", lw=.7); ax.set_xticks(range(3), ["rest\n(−1000 to −400 ms)", "reach\n(±200 ms)", "whole trial\n(−1000 to +2000 ms)"])
ax.set_ylabel("held-out gain of couplings, M0 − M1 surprise\n(bits per neuron per 10-ms bin)"); ax.set_title("Couplings improve exact held-out prediction", loc="left")
fig.text(.01, -.04, "One dot per session (21). Times from the 1-cm crossing. 5-fold CV over reaches. Exact likelihood (sessions > 16 cells: fixed 16-cell subset).", fontsize=7.5, color="#555")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"M1_vs_M0_exact.{e}"), dpi=200, bbox_inches="tight")
