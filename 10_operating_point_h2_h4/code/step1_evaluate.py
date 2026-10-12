"""Evaluate E10 step 1 against the criteria pre-stated in amendment 5. Usage: python step1_evaluate.py results/step1/step1_results.csv assets"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); m = x.mean(); h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h
rows = []
for (scen, N, R), g in df.groupby(["scen", "N", "R"]):
    r = dict(scen=scen, N=N, R=R, n_rep=len(g))
    if scen in ("A", "C"):
        m, lo, hi = ci(g.kappa_hat); r.update(metric="kappa_hat", mean=m, lo=lo, hi=hi, truth=0.6 if scen == "C" else 0.0,
                                               gain_pos=int((g.kappa_gain_reach > 0).sum()), gain_mean=g.kappa_gain_reach.mean())
        if R == 60:
            ok = (lo <= 0 <= hi and abs(m) < 0.15) if scen == "A" else (0.3 <= m <= 0.9 and lo > 0 and r["gain_pos"] >= 4)
            r["pass"] = bool(ok)
    else:
        m, lo, hi = ci(g.rho_diff); r.update(metric="rho_real_minus_null", mean=m, lo=lo, hi=hi, truth=np.nan,
                                              rho_real=g.est_neuron_rho.mean(), rho_null=g.null_neuron_rho.mean())
        if R == 60:
            r["pass"] = bool((abs(m) < 0.15 and lo <= 0 <= hi) if scen == "D" else (m < 0 and hi < 0))
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{out}/tables/step1_summary.csv", index=False)
gate = T[T["pass"].notna()]
print(T.round(3).to_string()); print("\nGATE (R=60):", "ALL PASS" if gate["pass"].all() else "FAIL"); print(gate[["scen", "N", "pass"]].to_string())
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, sc, key, title in ((ax[0], ("A", "C"), "kappa_hat", "kappa fit (A: truth 0, C: truth 0.6)"), (ax[1], ("D", "E"), "rho_diff", "rho_real - rho_null (D: 0, E: <0)")):
    for i, s in enumerate(sc):
        for j, (N, R) in enumerate([(11, 20), (11, 60), (15, 20), (15, 60)]):
            v = df[(df.scen == s) & (df.N == N) & (df.R == R)][key]
            a.scatter(np.full(len(v), i * 5 + j) , v, s=14)
        a.set_xticks([0, 1, 2, 3, 5, 6, 7, 8]); a.set_xticklabels([f"{s}\nN{N}\nR{R}" for s in sc for N, R in [(11, 20), (11, 60), (15, 20), (15, 60)]], fontsize=6)
    a.axhline(0, c="gray", lw=.5); a.set_title(title, fontsize=9)
ax[0].axhline(0.6, c="r", lw=.5)
fig.suptitle("E10 step 1: SYNTHETIC method validation of amendments 2-3", fontsize=10); fig.tight_layout()
fig.savefig(f"{out}/figures/step1_synthetic_amendments.png", dpi=150)
