"""E10 step 3: aggregate step-2 task pickles and run the pre-registered session-level statistics.
Usage: python step3_stats.py RESULTS_DIR ASSETS_DIR
Unit = session (n = 21): conditions (blocks) averaged within session; subsets averaged within condition (amendment 1).
Primary family {H2, H3, H4a} with BH-FDR (amendment 6). H4b descriptive only."""
import sys, glob, pickle, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
res, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
df = pd.DataFrame([pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{res}/task_*.pkl"))])
df["session"] = df.key.str.split("_").str[0]; df["block"] = df.key.str.split("_").str[1].astype(int)
df["cohort"] = np.where(df.session.str.startswith("21"), "2021", "2022")
# condition level: average over subsets
num = [c for c in df.columns if df[c].dtype.kind in "fiub" and c not in ("sub",)]
cond = df.groupby(["key", "session", "block", "cohort"], as_index=False)[num].mean()
cond["n_tasks"] = df.groupby("key").size().values
cond["dI_exceeds_null"] = cond.real_dI > cond.null_dI_p95
cond.to_csv(f"{out}/tables/e10_conditions.csv", index=False)
sess = cond.groupby(["session", "cohort"], as_index=False).mean(numeric_only=True)
sess.to_csv(f"{out}/tables/e10_sessions.csv", index=False)
sess["dI_net"] = sess.real_dI - sess.null_dI_mean

def wilcoxon(x, label):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    r = stats.wilcoxon(x); return dict(test=label, n=len(x), median=np.median(x), n_pos=int((x > 0).sum()), p=r.pvalue)

rows = [wilcoxon(sess.real_dI, "H2: dI/N reach-baseline (session means)"),
        wilcoxon(sess.dI_net, "H2: dI/N minus own shuffle-null mean"),
        wilcoxon(sess.kappa_hat, "H3b: kappa-hat (session means)")]
# H3a: empirical time-locked cov change vs fixed-J prediction
pred = sess.pred_dchiJ_folds; emp = sess.emp_dcov
a = stats.spearmanr(cond.emp_dcov, cond.pred_dchiJ_folds, nan_policy="omit")
rows.append(dict(test="H3a: Spearman(emp dcov, fold-predicted dchiJ), across conditions", n=len(cond), median=a.statistic, n_pos=int((np.sign(emp) == np.sign(pred)).sum()), p=a.pvalue))
b = stats.spearmanr(cond.emp_dcov, cond.real_dchiJ, nan_policy="omit")
rows.append(dict(test="H3a (full-data fit prediction): Spearman across conditions", n=len(cond), median=b.statistic, n_pos=np.nan, p=b.pvalue))
c = stats.spearmanr(cond.real_dv, cond.real_dI)
rows.append(dict(test="H4a: Spearman(dv, dI/N) across conditions", n=len(cond), median=c.statistic, n_pos=np.nan, p=c.pvalue))
d = stats.spearmanr(sess.real_dv, sess.real_dI)
rows.append(dict(test="H4a readout: Spearman(dv, dI/N) across sessions", n=len(sess), median=d.statistic, n_pos=np.nan, p=d.pvalue))
rows.append(wilcoxon(sess.real_dv, "H1 check: dv (p(1-p)) reach-baseline"))
rows.append(wilcoxon(sess.real_dchiJ, "secondary: dchiJ/N"))
rows.append(wilcoxon(sess.real_dC, "secondary: dC/N (heat capacity)"))
rows.append(wilcoxon(sess.real_rho - sess.null_rho_mean, "H4b EXPLORATORY (failed synthetic validation): rho_real - rho_null"))
T = pd.DataFrame(rows)
# BH-FDR over primary family {H2, H3, H4a}
prim = {"H2": "H2: dI/N reach-baseline (session means)", "H3b": "H3b: kappa-hat (session means)", "H3a": "H3a: Spearman(emp dcov, fold-predicted dchiJ), across conditions",
        "H4a": "H4a: Spearman(dv, dI/N) across conditions"}
fam = {"H2": T.loc[T.test == prim["H2"], "p"].iloc[0], "H3a": T.loc[T.test == prim["H3a"], "p"].iloc[0], "H3b": T.loc[T.test == prim["H3b"], "p"].iloc[0], "H4a": T.loc[T.test == prim["H4a"], "p"].iloc[0]}
# amendment 6: family {H2, H3, H4a}; H3 = both H3a and H3b must hold, represented by the larger p
fam3 = {"H2": fam["H2"], "H3": max(fam["H3a"], fam["H3b"]), "H4a": fam["H4a"]}
ps = np.array(list(fam3.values())); o = np.argsort(ps); q = np.empty(3); m = 3
q[o] = np.minimum.accumulate((ps[o] * m / (np.arange(m) + 1))[::-1])[::-1]
T["p_FDR_primary"] = np.nan
for (k, v), qq in zip(fam3.items(), q):
    for t in {"H2": [prim["H2"]], "H3": [prim["H3a"], prim["H3b"]], "H4a": [prim["H4a"]]}[k]: T.loc[T.test == t, "p_FDR_primary"] = qq
# cohort splits
for co, g in sess.groupby("cohort"):
    T.loc[len(T)] = dict(**wilcoxon(g.real_dI, f"H2 within cohort {co}: dI/N"), p_FDR_primary=np.nan)
    T.loc[len(T)] = dict(**wilcoxon(g.real_dv, f"H1 within cohort {co}: dv"), p_FDR_primary=np.nan)
# mouse effect not available in spikes_v2.npz (no mouse id): report session means; mouse column added in RESULTS if the mapping is supplied
T.to_csv(f"{out}/tables/e10_tests.csv", index=False)
print(T.round(4).to_string())
print("\nsessions exceeding own null (dI > p95 null):", int((cond.groupby('session').dI_exceeds_null.mean() >= 0.5).sum()), "of", sess.shape[0])
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(sess.real_dv, sess.real_dI, c=(sess.cohort == "2022"), cmap="coolwarm"); ax[0].axhline(0, c="gray", lw=.5); ax[0].axvline(0, c="gray", lw=.5)
ax[0].set_xlabel("session mean Δ p(1-p)  (reach − baseline)"); ax[0].set_ylabel("Δ I/N (bits/neuron)"); ax[0].set_title("H4a: operating point vs coupling influence", fontsize=9)
ax[1].scatter(cond.pred_dchiJ_folds, cond.emp_dcov, s=12); ax[1].plot([cond.pred_dchiJ_folds.min(), cond.pred_dchiJ_folds.max()], [cond.pred_dchiJ_folds.min(), cond.pred_dchiJ_folds.max()], c="gray", lw=.5)
ax[1].set_xlabel("fixed-J predicted Δχ_J/N"); ax[1].set_ylabel("empirical time-locked Δcov/N"); ax[1].set_title("H3a", fontsize=9)
ax[2].hist(sess.kappa_hat, bins=12); ax[2].axvline(0, c="r", lw=.7); ax[2].set_xlabel("κ̂ (session means)"); ax[2].set_title("H3b: coupling scale in reach window", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/figures/e10_h2_h4.png", dpi=150)
