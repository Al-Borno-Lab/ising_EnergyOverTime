"""E10 step 3: aggregate step-2 task pickles and run the pre-registered session-level statistics.
Usage: python step3_stats.py RESULTS_DIR ASSETS_DIR [--synthetic-ok]
Decision logic follows PREREGISTRATION_OPERATING_POINT.md amendments 6 and 7 (written before any result was read).
Unit = session (n = 21); subsets averaged within condition, conditions within session. Primary family {H2, H3, H4a}, BH-FDR.
H3 holds only if H3a is supported AND H3b shows no change (a significant kappa-hat or held-out gain REFUTES fixed J)."""
import sys, glob, pickle, os, warnings, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
res, out = sys.argv[1], sys.argv[2]; PARTDIR = sys.argv[3] if len(sys.argv) > 3 else None
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
complete = set(EXP.key[[int(cnt.get(k, 0)) == n for k, n in zip(EXP.key, EXP.n_expected_tasks)]])
KEYCOLS = ["real_dI", "real_dchiJ", "real_dv", "emp_dcov", "kappa_hat", "kappa_gain_reach", "pred_dchiJ_folds", "null_dI_p95", "null_dI_mean"]
for k, g in df.groupby("key"):
    bad = [c for c in KEYCOLS if g[c].isna().any()]
    if bad:
        warn(f"condition {k}: NaN in {bad} in {int(g[bad].isna().any(axis=1).sum())} tasks; marked incomplete"); complete.discard(k)
over = [k for k in cnt.index if k in set(EXP.key) and cnt[k] > int(EXP.set_index("key").n_expected_tasks[k])]
for k in over: warn(f"condition {k}: more tasks than expected ({cnt[k]})")
df["task_ok"] = df.converged & df.kappa_converged
num = [c for c in df.columns if df[c].dtype.kind in "fiub" and c not in ("sub",)]
cond = df.groupby("key", as_index=False)[num].mean()
cond["n_tasks"] = df.groupby("key").size().values
cond["all_converged"] = df.groupby("key").task_ok.all().values
cond = cond.merge(EXP[["key", "session", "block", "mouse"]], on="key"); cond["cohort"] = np.where(cond.session.str.startswith("21"), "2021", "2022")
cond["complete"] = cond.key.isin(complete)
cond.to_csv(f"{out}/tables/e10_conditions.csv", index=False)
conv = df.assign(cohort=np.where(df.key.str.startswith("21"), "2021", "2022")).groupby("cohort").agg(n_tasks=("key", "size"), real_fit_not_converged=("converged", lambda x: int((~x).sum())),
        kappa_fit_not_converged=("kappa_converged", lambda x: int((~x).sum())))
conv.to_csv(f"{out}/tables/e10_convergence_by_cohort.csv"); print(conv.to_string())
sess_n = cond.groupby("session").complete.sum()
for s, n in sess_n.items():
    if n < 3: warn(f"session {s}: only {n}/3 complete conditions; session mean uses incomplete data")
N_SESS_EXPECTED = EXP.session.nunique()

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
            warnings.simplefilter("ignore"); m = smf.mixedlm(formula, data, groups=data[group], missing="drop").fit(reml=True)
        return {k: (float(m.params[k]), float(m.pvalues[k])) for k in m.params.index if k != "Group Var"} | {"group_var": float(m.cov_re.iloc[0, 0])}
    except Exception as e:
        warn(f"mixed model failed ({formula}): {e}"); return {}

def bh(ps):
    ps = np.asarray(ps, float)
    if np.isnan(ps).any(): warn("NaN p-value entered BH; treated as 1 so that the other q-values remain defined"); ps = np.where(np.isnan(ps), 1.0, ps)
    m = len(ps); o = np.argsort(ps); q = np.empty(m)
    q[o] = np.minimum(1.0, np.minimum.accumulate((ps[o] * m / (np.arange(m) + 1))[::-1])[::-1]); return q

def analyse(cd, tag):
    """Run all pre-registered tests on a condition table cd. Returns (rows, verdict dict)."""
    sess = cd.groupby(["session", "cohort", "mouse"], as_index=False).mean(numeric_only=True)
    R = {}
    # H2
    h2 = wil(sess.real_dI); n_s = len(sess)
    # A7 primary bound: mean over a session's conditions (and subsets) of the per-task 95th percentile (conservative; ~4-5x too strict for N > 20)
    bound = cd.groupby("session").null_dI_p95.mean()
    # sensitivity: Gaussian approximation of the session-averaged null (sd of the mean = sqrt(sum sd_task^2)/n)
    sd_t = (cd.null_dI_p95 - cd.null_dI_mean) / 1.645; g = cd.assign(sd2=sd_t ** 2 / cd.n_tasks).groupby("session")
    bound_g = g.null_dI_mean.mean() + 1.645 * np.sqrt(g.sd2.sum()) / g.size()
    sidx = sess.set_index("session"); above = (sidx.real_dI > bound); above_g = (sidx.real_dI > bound_g)
    sess["null_bound"] = sess.session.map(bound); sess["above_null"] = sess.session.map(above); sess["null_bound_gauss"] = sess.session.map(bound_g); sess["above_null_gauss"] = sess.session.map(above_g)
    exc = int(above.sum()); h2["n_above_null"] = exc; h2["n_above_null_gauss_sens"] = int(above_g.sum())
    h2["p_above_null"] = float(stats.binomtest(exc, n_s, 0.05, alternative="greater").pvalue)   # reported; NOT the gate (A7 addendum)
    h2["above_null_by_cohort"] = "; ".join(f"{c}:{int(above[sess.set_index('session').cohort == c].sum())}/{int((sess.cohort == c).sum())}" for c in sorted(sess.cohort.unique()))
    h2["gate_majority_above_null"] = bool(exc > n_s / 2)
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
    gx = sess.kappa_gain_reach.dropna().values
    R["H3b_gain"]["p_greater"] = float(stats.wilcoxon(gx, alternative="greater").pvalue) if len(gx) >= 6 and not np.allclose(gx, 0) else np.nan
    # H4a
    sp = spear(cd.real_dv, cd.real_dI, f"{tag} H4a"); R["H4a"] = dict(rho=sp["rho"], p=sp["p"], n=sp["n"])
    mm = mixed("real_dI ~ real_dv", cd, "session"); R["H4a"]["mixed_slope"], R["H4a"]["mixed_p"] = mm.get("real_dv", (np.nan, np.nan))
    sps = spear(sess.real_dv, sess.real_dI, f"{tag} H4a session"); R["H4a_session_readout"] = dict(rho=sps["rho"], p=sps["p"], n=sps["n"])
    R["H1_dv"] = wil(sess.real_dv); R["secondary_dchiJ"] = wil(sess.real_dchiJ); R["secondary_dC"] = wil(sess.real_dC)
    R["H4b_exploratory_FAILED_VALIDATION"] = wil(sess.real_rho - sess.null_rho_mean)
    # verdicts (amendment 7)
    q = bh([R["H2"]["p"], R["H3a"]["p"], R["H4a"]["p"]])
    h2_ok = q[0] < .05 and R["H2"]["median"] > 0 and R["H2"]["gate_majority_above_null"]
    h3a_ok = R["H3a"]["rho"] > 0 and q[1] < .05 and R["H3a"]["p_sign"] < .05 and R["H3a"]["sessions_sign_agree"] > n_s / 2
    # H3b (A7 addendum): kappa-hat two-sided (a change in either direction refutes fixed J); held-out gain one-sided (only an IMPROVEMENT refutes)
    h3b_eval = not (np.isnan(R["H3b_kappa"]["p"]) or np.isnan(R["H3b_gain"]["p_greater"]))
    if not h3b_eval: warn(f"{tag}: H3b not evaluable (NaN p for kappa-hat or held-out gain)")
    h3b_ok = bool(h3b_eval and R["H3b_kappa"]["p"] > .05 and not (R["H3b_gain"]["p_greater"] < .05))
    h3_ok = bool(h3a_ok and h3b_ok); h4a_ok = q[2] < .05 and R["H4a"]["rho"] > 0
    V = dict(H3b_evaluable=bool(h3b_eval), n_sessions=n_s, n_conditions_complete=int(cd.complete.sum()), n_warnings=len(WARN), q_H2=q[0], q_H3a=q[1], q_H4a=q[2], H2_supported=bool(h2_ok), H3a_supported=bool(h3a_ok), H3b_no_change=bool(h3b_ok),
             H3_holds=h3_ok, H4a_supported=bool(h4a_ok), OVERALL_SUPPORTED=bool(h2_ok and h3_ok and h4a_ok))
    if not V["OVERALL_SUPPORTED"]:
        why = []
        if R["H2"]["median"] <= 0: why.append("dI <= 0")
        elif q[0] < .05 and not R["H2"]["gate_majority_above_null"]: why.append("dI > 0 but not above null in a majority of sessions")
        elif not h2_ok: why.append("H2 not significant")
        if not h3a_ok: why.append("H3a not supported (empirical covariance does not change as fixed-J predicts)")
        if not h3b_eval: why.append("H3b not evaluable (NaN p)")
        elif not h3b_ok: why.append("H3b: kappa-hat or held-out gain shows a coupling change (fixed J refuted)")
        if not h4a_ok: why.append("H4a not supported (dI unrelated to dv or wrong sign)")
        V["refuted_because"] = "; ".join(why)
    rows = []
    for k, v in R.items(): rows.append({"set": tag, "test": k, **{kk: vv for kk, vv in v.items()}})
    rows.append({"set": tag, "test": "VERDICT", **V})
    return rows, V, sess

rows, V, sess = analyse(cond, "primary")
if PARTDIR:   # exact session-level null for sessions whose conditions all have per-shuffle values (the six N = 19 conditions: 220515, 220516)
    ex = []
    for s_id, g in cond.groupby("session"):
        nul = []
        for k in g.key:
            parts = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{PARTDIR}/part_{k}_-1_[0-3].pkl"))]
            if len(parts) != 4: nul = None; break
            nul.append({b: dI for p in parts for (b, dI, _, _) in p["null"]})
        if nul and all(len(n) == 20 for n in nul):
            m = np.mean([[n[b] for b in range(20)] for n in nul], axis=0); real = sess.set_index("session").real_dI[s_id]
            ex.append(dict(session=s_id, real_dI=real, exact_null_p95=float(np.percentile(m, 95)), exact_above_null=bool(real > np.percentile(m, 95)), n_shuffles=20))
    pd.DataFrame(ex).to_csv(f"{out}/tables/e10_exact_null_n19.csv", index=False); print("exact N=19 session nulls:", ex)
# cohort rule (section 6; A7 second addendum): based on the cohort that has power.
#   absent in 2021 = 2021-only H2 Wilcoxon median <= 0 OR p >= 0.05;  present in 2022 = 2022 median dI/N > 0 (direction only; n = 6).
#   If 2021 is absent and 2022 present, the claim is "not established outside 2022" -> REFUTED. Conservative: an effect present in both cohorts
#   whose 2021 part alone misses p < 0.05 is also refuted. 2021-only and 2022-only statistics are always reported.
cohort_V, coh = {}, {}
for co in ("2021", "2022"):
    cc = cond[cond.cohort == co]
    if cc.session.nunique() >= 6:
        r, Vc, _ = analyse(cc, f"cohort_{co}"); rows += r; cohort_V[co] = Vc
        h = [x for x in r if x["test"] == "H2"][0]; coh[co] = h
    else:
        warn(f"cohort {co}: fewer than 6 sessions, cohort rule cannot be evaluated"); coh[co] = None
present = {co: (None if h is None else bool(h["median"] > 0 and h["p"] < .05)) for co, h in coh.items()}
absent_2021 = None if coh["2021"] is None else bool(coh["2021"]["median"] <= 0 or not (coh["2021"]["p"] < .05))
present_2022 = None if coh["2022"] is None else bool(coh["2022"]["median"] > 0)
V["H2_2021_median"] = None if coh["2021"] is None else coh["2021"]["median"]; V["H2_2021_p"] = None if coh["2021"] is None else coh["2021"]["p"]
V["H2_2022_median"] = None if coh["2022"] is None else coh["2022"]["median"]; V["H2_2022_p"] = None if coh["2022"] is None else coh["2022"]["p"]
V["H2_absent_2021"], V["H2_present_2022_direction"] = absent_2021, present_2022
V["rule_2022_only_triggered"] = bool(absent_2021 and present_2022)
if V["rule_2022_only_triggered"]:
    V["OVERALL_SUPPORTED"] = False; V["refuted_because"] = (V.get("refuted_because", "") + "; " if V.get("refuted_because") else "") + "not established outside the 2022 cohort (section 6): 2021 alone absent, 2022 median > 0"
V["incomplete"] = bool(V["n_sessions"] < N_SESS_EXPECTED or V["n_conditions_complete"] < len(EXP) or coh["2021"] is None or coh["2022"] is None)
V["VERDICT_STATUS"] = "INCOMPLETE (do not interpret)" if V["incomplete"] else ("SUPPORTED" if V["OVERALL_SUPPORTED"] else "REFUTED/NOT SUPPORTED")
for row in rows:
    if row["set"] == "primary" and row["test"] == "VERDICT": row.update(V)
rows += analyse(cond[cond.N_full >= 5], "sens_N_ge_5")[0]
rows += analyse(cond[cond.all_converged], "sens_converged_only")[0]
rows += analyse(cond[cond.complete], "sens_complete_conditions_only")[0]
T = pd.DataFrame(rows); T.to_csv(f"{out}/tables/e10_tests.csv", index=False)
sess.to_csv(f"{out}/tables/e10_sessions.csv", index=False)
open(f"{out}/tables/e10_warnings.txt", "w").write("\n".join(WARN) + "\n")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T[T.set == "primary"].round(4).to_string()); print("\nVERDICT:", V)
print("cohort H2: 2021 median/p", V["H2_2021_median"], V["H2_2021_p"], "| 2022 median/p", V["H2_2022_median"], V["H2_2022_p"], "| cohort verdicts H2_supported:", {co: (v and v["H2_supported"]) for co, v in cohort_V.items()})
print(f"\n{len(WARN)} warnings (see e10_warnings.txt)")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(sess.real_dv, sess.real_dI, c=(sess.cohort == "2022"), cmap="coolwarm"); ax[0].axhline(0, c="gray", lw=.5); ax[0].axvline(0, c="gray", lw=.5)
ax[0].set_xlabel("session mean Δ p(1-p)  (reach − baseline)"); ax[0].set_ylabel("Δ I/N (bits/neuron)"); ax[0].set_title("H4a readout: operating point vs coupling influence", fontsize=9)
lim = [min(cond.real_dchiJ.min(), cond.emp_dcov.min()), max(cond.real_dchiJ.max(), cond.emp_dcov.max())]
ax[1].scatter(cond.real_dchiJ, cond.emp_dcov, s=12); ax[1].plot(lim, lim, c="gray", lw=.5)
ax[1].set_xlabel("fixed-J predicted Δχ_J/N (full-data fit)"); ax[1].set_ylabel("empirical time-locked Δcov/N"); ax[1].set_title("H3a", fontsize=9)
ax[2].hist(sess.kappa_hat, bins=12); ax[2].axvline(0, c="r", lw=.7); ax[2].set_xlabel("κ̂ (session means)"); ax[2].set_title("H3b: coupling scale in reach window", fontsize=9)
fig.tight_layout(); fig.savefig(f"{out}/figures/e10_h2_h4.png", dpi=150)
