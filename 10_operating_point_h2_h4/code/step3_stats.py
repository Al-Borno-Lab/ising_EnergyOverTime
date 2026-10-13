"""E10 step 3: aggregate step-2 task pickles and run the pre-registered session-level statistics.
Usage: python step3_stats.py RESULTS_DIR ASSETS_DIR [--synthetic-ok]
Decision logic follows PREREGISTRATION_OPERATING_POINT.md amendments 6 and 7 (written before any result was read).
Unit = session (n = 21); subsets averaged within condition, conditions within session. Primary family {H2, H3, H4a}, BH-FDR.
H3 holds only if H3a is supported AND H3b shows no change (a significant kappa-hat or held-out gain REFUTES fixed J)."""
import sys, glob, pickle, os, warnings, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
res, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/tables", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
EXP = pd.read_csv(f"{out}/tables/expected_conditions.csv", dtype={"key": str, "session": str})
WARN = []
def warn(msg): WARN.append(msg); print("WARNING:", msg)

# ---------------- load and completeness ----------------
df = pd.DataFrame([pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{res}/task_*.pkl"))])
cnt = df.groupby("key").size()
for _, r in EXP.iterrows():
    n = int(cnt.get(r.key, 0))
    if n < r.n_expected_tasks: warn(f"condition {r.key}: {n}/{r.n_expected_tasks} tasks present")
extra = set(df.key) - set(EXP.key)
if extra: warn(f"unexpected conditions {sorted(extra)}")
nonconv = df[~(df.converged & df.kappa_converged)]
for _, r in nonconv.iterrows(): warn(f"non-converged: {r.key} sub {r['sub']} (real fit {r.converged}, kappa fit {r.kappa_converged})")
nan_cols = [c for c in df.columns if df[c].dtype.kind == "f" and df[c].isna().any()]
for c in nan_cols: warn(f"NaN in column {c}: {df[c].isna().sum()} tasks")
complete = set(EXP.key[[int(cnt.get(k, 0)) >= n for k, n in zip(EXP.key, EXP.n_expected_tasks)]])
df["task_ok"] = df.converged & df.kappa_converged
num = [c for c in df.columns if df[c].dtype.kind in "fiub" and c not in ("sub",)]
cond = df.groupby("key", as_index=False)[num].mean()
cond["n_tasks"] = df.groupby("key").size().values
cond["all_converged"] = df.groupby("key").task_ok.all().values
cond = cond.merge(EXP[["key", "session", "block", "mouse"]], on="key"); cond["cohort"] = np.where(cond.session.str.startswith("21"), "2021", "2022")
cond["complete"] = cond.key.isin(complete)
cond.to_csv(f"{out}/tables/e10_conditions.csv", index=False)
sess_n = cond.groupby("session").complete.sum()
for s, n in sess_n.items():
    if n < 3: warn(f"session {s}: only {n}/3 complete conditions; session mean uses incomplete data")

# ---------------- helpers ----------------
def wil(x):
    x = np.asarray(x, float); n_all = len(x); x = x[~np.isnan(x)]
    if len(x) < n_all: warn(f"wilcoxon: dropped {n_all - len(x)} NaN of {n_all}")
    if len(x) < 6 or np.allclose(x, 0): return dict(n=len(x), median=np.median(x) if len(x) else np.nan, n_pos=int((x > 0).sum()), p=np.nan, effect_r=np.nan)
    r = stats.wilcoxon(x); rk = stats.rankdata(np.abs(x)); wp, wm = rk[x > 0].sum(), rk[x < 0].sum()
    return dict(n=len(x), median=float(np.median(x)), n_pos=int((x > 0).sum()), p=float(r.pvalue), effect_r=float((wp - wm) / (wp + wm)))

def spear(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b))
    if (~ok).sum(): warn(f"{label}: dropped {(~ok).sum()} NaN pairs")
    r = stats.spearmanr(a[ok], b[ok]); return dict(n=int(ok.sum()), rho=float(r.statistic), p=float(r.pvalue))

def sign_test(n_agree, n):
    return float(stats.binomtest(n_agree, n, 0.5).pvalue) if n else np.nan

def mixed(formula, data, group):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m = smf.mixedlm(formula, data, groups=data[group]).fit(reml=True)
        return {k: (float(m.params[k]), float(m.pvalues[k])) for k in m.params.index if k != "Group Var"} | {"group_var": float(m.cov_re.iloc[0, 0])}
    except Exception as e:
        warn(f"mixed model failed ({formula}): {e}"); return {}

def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m)
    q[o] = np.minimum(1.0, np.minimum.accumulate((ps[o] * m / (np.arange(m) + 1))[::-1])[::-1]); return q

def analyse(cd, tag):
    """Run all pre-registered tests on a condition table cd. Returns (rows, verdict dict)."""
    sess = cd.groupby(["session", "cohort", "mouse"], as_index=False).mean(numeric_only=True)
    R = {}
    # H2
    h2 = wil(sess.real_dI); bound = cd.groupby("session").null_dI_p95.mean()
    exc = (sess.set_index("session").real_dI > bound).sum(); n_s = len(sess)
    h2["n_above_null"] = int(exc); h2["p_above_null"] = float(stats.binomtest(int(exc), n_s, 0.05, alternative="greater").pvalue)
    mm = mixed("real_dI ~ 1", sess, "mouse"); h2["mixed_mouse_intercept"], h2["mixed_mouse_p"] = mm.get("Intercept", (np.nan, np.nan))
    R["H2"] = h2
    # H3a primary (full-data prediction); sensitivity (fold-averaged)
    for nm, col in (("H3a", "real_dchiJ"), ("H3a_sens_folds", "pred_dchiJ_folds")):
        sp = spear(cd.emp_dcov, cd[col], f"{tag} {nm}"); agree = int((np.sign(sess.emp_dcov) == np.sign(sess[col])).sum())
        e_pred = (sess.emp_dcov - sess[col]).abs(); e_frozen = sess.emp_dcov.abs()     # frozen-field reference: predicted change = 0
        ref = wil(e_frozen - e_pred)
        R[nm] = dict(rho=sp["rho"], p=sp["p"], n=sp["n"], sessions_sign_agree=agree, n_sessions=len(sess), p_sign=sign_test(agree, len(sess)),
                     median_abs_err_gain_vs_frozen=ref["median"], p_vs_frozen=ref["p"])
    # H3b
    R["H3b_kappa"] = wil(sess.kappa_hat); R["H3b_gain"] = wil(sess.kappa_gain_reach)
    # H4a
    sp = spear(cd.real_dv, cd.real_dI, f"{tag} H4a"); R["H4a"] = dict(rho=sp["rho"], p=sp["p"], n=sp["n"])
    mm = mixed("real_dI ~ real_dv", cd, "session"); R["H4a"]["mixed_slope"], R["H4a"]["mixed_p"] = mm.get("real_dv", (np.nan, np.nan))
    sps = spear(sess.real_dv, sess.real_dI, f"{tag} H4a session"); R["H4a_session_readout"] = dict(rho=sps["rho"], p=sps["p"], n=sps["n"])
    R["H1_dv"] = wil(sess.real_dv); R["secondary_dchiJ"] = wil(sess.real_dchiJ); R["secondary_dC"] = wil(sess.real_dC)
    R["H4b_exploratory_FAILED_VALIDATION"] = wil(sess.real_rho - sess.null_rho_mean)
    # verdicts (amendment 7)
    q = bh([R["H2"]["p"], R["H3a"]["p"], R["H4a"]["p"]])
    h2_ok = q[0] < .05 and R["H2"]["median"] > 0 and R["H2"]["p_above_null"] < .05
    h3a_ok = R["H3a"]["rho"] > 0 and R["H3a"]["p"] < .05 and R["H3a"]["p_sign"] < .05 and R["H3a"]["sessions_sign_agree"] > n_s / 2
    h3b_ok = (R["H3b_kappa"]["p"] > .05) and (R["H3b_gain"]["p"] > .05)
    h3_ok = bool(h3a_ok and h3b_ok); h4a_ok = q[2] < .05 and R["H4a"]["rho"] > 0
    V = dict(q_H2=q[0], q_H3a=q[1], q_H4a=q[2], H2_supported=bool(h2_ok), H3a_supported=bool(h3a_ok), H3b_no_change=bool(h3b_ok),
             H3_holds=h3_ok, H4a_supported=bool(h4a_ok), OVERALL_SUPPORTED=bool(h2_ok and h3_ok and h4a_ok))
    if not V["OVERALL_SUPPORTED"]:
        why = []
        if R["H2"]["median"] <= 0: why.append("dI <= 0")
        elif not h2_ok: why.append("H2 not significant/above null")
        if not h3a_ok: why.append("H3a not supported (empirical covariance does not change as fixed-J predicts)")
        if not h3b_ok: why.append("H3b: kappa-hat or held-out gain shows a coupling change (fixed J refuted)")
        if not h4a_ok: why.append("H4a not supported (dI unrelated to dv or wrong sign)")
        V["refuted_because"] = "; ".join(why)
    rows = []
    for k, v in R.items(): rows.append({"set": tag, "test": k, **{kk: vv for kk, vv in v.items()}})
    rows.append({"set": tag, "test": "VERDICT", **V})
    return rows, V, sess

rows, V, sess = analyse(cond, "primary")
cohort_V = {}
for co in ("2021", "2022"):
    r, Vc, _ = analyse(cond[cond.cohort == co], f"cohort_{co}") if cond[cond.cohort == co].session.nunique() >= 6 else ([], None, None)
    rows += r; cohort_V[co] = Vc
rows += analyse(cond[cond.N_full >= 5], "sens_N_ge_5")[0]
rows += analyse(cond[cond.all_converged], "sens_converged_only")[0]
rows += analyse(cond[cond.complete], "sens_complete_conditions_only")[0]
T = pd.DataFrame(rows); T.to_csv(f"{out}/tables/e10_tests.csv", index=False)
sess.to_csv(f"{out}/tables/e10_sessions.csv", index=False)
open(f"{out}/tables/e10_warnings.txt", "w").write("\n".join(WARN) + "\n")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T[T.set == "primary"].round(4).to_string()); print("\nVERDICT:", V)
if V["OVERALL_SUPPORTED"] is False:
    c = {co: (v and v["H2_supported"]) for co, v in cohort_V.items()}; print("per-cohort H2:", c)
print(f"\n{len(WARN)} warnings (see e10_warnings.txt)")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(sess.real_dv, sess.real_dI, c=(sess.cohort == "2022"), cmap="coolwarm"); ax[0].axhline(0, c="gray", lw=.5); ax[0].axvline(0, c="gray", lw=.5)
ax[0].set_xlabel("session mean Δ p(1-p)  (reach − baseline)"); ax[0].set_ylabel("Δ I/N (bits/neuron)"); ax[0].set_title("H4a readout: operating point vs coupling influence", fontsize=9)
lim = [min(cond.real_dchiJ.min(), cond.emp_dcov.min()), max(cond.real_dchiJ.max(), cond.emp_dcov.max())]
ax[1].scatter(cond.real_dchiJ, cond.emp_dcov, s=12); ax[1].plot(lim, lim, c="gray", lw=.5)
ax[1].set_xlabel("fixed-J predicted Δχ_J/N (full-data fit)"); ax[1].set_ylabel("empirical time-locked Δcov/N"); ax[1].set_title("H3a", fontsize=9)
ax[2].hist(sess.kappa_hat, bins=12); ax[2].axvline(0, c="r", lw=.7); ax[2].set_xlabel("κ̂ (session means)"); ax[2].set_title("H3b: coupling scale in reach window", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/figures/e10_h2_h4.png", dpi=150)
