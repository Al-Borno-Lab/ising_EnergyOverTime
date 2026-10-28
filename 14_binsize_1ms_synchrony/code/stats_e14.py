"""E14 statistics (design: PREREGISTRATION_2026-10-27.md).
    python stats_e14.py synth SYNTH_DIR ASSETS_DIR
    python stats_e14.py real  REAL_DIR  ASSETS_DIR"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODE, SRC, ASSETS = sys.argv[1], sys.argv[2], sys.argv[3]
TAB, FIG = os.path.join(ASSETS, "tables"), os.path.join(ASSETS, "figures"); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NBOOT, SEED = 10_000, 14
WINDOWS_PREREG = "{'B': ('cross', -1000, -400), 'E': ('end', -100, 0), 'A': ('cross', -90, 0)}"

def hl(x):
    x = np.asarray(x, float); i, j = np.triu_indices(len(x)); return float(np.median((x[i] + x[j]) / 2))

def boot(x):
    rng = np.random.default_rng(SEED); x = np.asarray(x, float)
    b = np.array([hl(x[rng.integers(0, len(x), len(x))]) for _ in range(NBOOT)])
    return {lv: tuple(np.percentile(b, [(100 - lv) / 2, 100 - (100 - lv) / 2])) for lv in (95, 90)}

def wil(x):
    x = np.asarray(x, float); return float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x != 0) else float("nan")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out

def row(x, **lab):
    x = pd.Series(x).dropna(); ci = boot(x.values) if len(x) >= 2 else {95: (np.nan,) * 2, 90: (np.nan,) * 2}
    return dict(**lab, n=len(x), median=float(x.median()), hl=hl(x.values) if len(x) else np.nan, ci95_lo=ci[95][0], ci95_hi=ci[95][1],
                ci90_lo=ci[90][0], ci90_hi=ci[90][1], n_pos=int((x > 0).sum()), p=wil(x.values))

def real():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "e14_[0-9]*.csv")))], ignore_index=True)
    assert (d.windows == WINDOWS_PREREG).all() and d.session.nunique() == 21, d.session.nunique()
    d.to_csv(os.path.join(TAB, "e14_rows.csv"), index=False)
    n1 = d[(d.kind == "N1") & (d.stim == 0) & (d.window == "E")].set_index("session")
    n2 = d[(d.kind == "N2") & (d.stim == 0)]
    k1 = n2[(n2.bin_ms == 1) & (n2.window == "E")].set_index("session")
    n3 = d[d.kind == "N3"].set_index("session"); resid = (n3.D_supp_obs - n3.D_supp_pred)
    # ---- primary family
    prim = [row(n1.D_supp, test="N1", stat="D_supp (SI change, suppressed pairs, stim0, E, 1 ms)"),
            row(k1.kappa_supp, test="N2", stat="kappa_supp (1 ms, E)"),
            row(resid, test="N3", stat="D_supp_obs - D_supp_pred (1 ms)")]
    assert all(np.isfinite(r["p"]) for r in prim), [r["p"] for r in prim]
    for r, q in zip(prim, bh([r["p"] for r in prim])): r["q_bh"] = q
    m = 0.5 * float(n1.D_supp.abs().median()); r3 = prim[2]; r3["equiv_margin"] = m
    r3["equiv"] = bool(-m < r3["ci90_lo"] and r3["ci90_hi"] < m)
    T = {r["test"]: r for r in prim}
    # ---- outcome (pre-registered decision table)
    n1_sig = T["N1"]["q_bh"] < 0.05 and T["N1"]["hl"] > 0
    k10 = n2[(n2.bin_ms == 10) & (n2.window == "E")].kappa_supp.dropna()
    k10_sig = wil(k10.values) < 0.05                       # unadjusted Wilcoxon (amendment A1)
    if not (T["N1"]["q_bh"] < 0.05): outcome = "N1 not significant: Nashef rise not reproduced with this window/class proxy"
    elif not n1_sig: outcome = "N1 significant with D < 0 (synchrony falls): report"
    elif r3["equiv"]: outcome = "(b) operating point: fixed J with moving fields reproduces the synchrony rise"
    elif T["N2"]["q_bh"] < 0.05 and T["N2"]["hl"] > 0 and not k10_sig: outcome = "(a) timescale: coupling changes at 1 ms but not at 10 ms"
    else: outcome = "mixed: report CIs"
    # ---- secondary / sensitivity
    sec = []
    for st in (0, 1, 2):
        for w in ("E", "A"):
            x = d[(d.kind == "N1") & (d.stim == st) & (d.window == w)]
            for c in ("D_supp", "D_other", "D_all"):
                if not (st == 0 and w == "E" and c == "D_supp"): sec.append(row(x[c], test="N1", stat=c, stim=st, window=w))
    for b in (1, 5, 10, 20):
        for w in ("E", "A"):
            x = n2[(n2.bin_ms == b) & (n2.window == w)]
            for c in ("kappa_supp", "kappa_other"):
                if not (b == 1 and w == "E" and c == "kappa_supp"): sec.append(row(x[c], test="N2", stat=c, bin_ms=b, window=w))
    kk = n2[n2.window == "E"].pivot(index="session", columns="bin_ms", values="kappa_supp")
    sec.append(row(kk[1] - kk[10], test="N2", stat="kappa_supp(1 ms) - kappa_supp(10 ms)", window="E"))
    sec.append(row(n3.D_all_obs - n3.D_all_pred, test="N3", stat="D_all_obs - D_all_pred"))
    sec.append(row(n1.D_supp - n1.D_other, test="selection", stat="D_supp - D_other (stim0, E, 1 ms)"))          # (c), amendment A1
    sec.append(row(k1.kappa_supp - k1.kappa_other, test="selection", stat="kappa_supp - kappa_other (1 ms, E)"))
    frac = (n3.D_supp_pred / n3.D_supp_obs).replace([np.inf, -np.inf], np.nan)
    for name, keep in (("cohort2021", lambda s: s < 220000), ("cohort2022", lambda s: s >= 220000), ("without_210511", lambda s: s != 210511)):
        for r_, (t, x) in zip(prim, (("N1", n1.D_supp), ("N2", k1.kappa_supp), ("N3", resid))):
            sec.append(row(x[keep(x.index)], test=t, stat=r_["stat"], subset=name))
    tests = pd.DataFrame(prim + sec); tests.to_csv(os.path.join(TAB, "e14_tests.csv"), index=False)
    conv = dict(kappa_not_converged=int((~n2.kappa_converged.astype(bool)).sum()), m1_not_converged=int((~n2.m1_converged.astype(bool)).sum()), n2_rows=len(n2))
    summ = dict(outcome=outcome, n_sessions=int(d.session.nunique()), n1_sessions_with_supp_pairs=int(n1.D_supp.notna().sum()),
                fraction_explained_median=float(frac.median()), convergence=conv,
                primary={r["test"]: {k: r[k] for k in r if k not in ("test",)} for r in prim}, kappa10_supp_p=wil(k10.values))
    json.dump(summ, open(os.path.join(TAB, "e14_summary.json"), "w"), indent=1, default=float); print(json.dumps(summ, indent=1, default=float))
    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (lab, x) in enumerate((("suppressed pairs", n1.D_supp), ("other pairs", n1.D_other))):
        v = x.dropna().values; ax[0].scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=14, color="#1f4e99", alpha=.7)
        ax[0].plot([i - .25, i + .25], [np.median(v)] * 2, color="k", lw=2)
    ax[0].axhline(0, color="#999", lw=.7); ax[0].set_xticks([0, 1], ["suppressed\npairs", "other\npairs"]); ax[0].set_ylabel("change in shuffle-corrected SI\n(100 ms before endpoint minus baseline)")
    ax[0].set_title("a  N1: 1-ms synchrony (stim 0)", loc="left")
    for c, col in (("kappa_supp", "#c0392b"), ("kappa_other", "#888")):
        M = n2[n2.window == "E"].pivot(index="session", columns="bin_ms", values=c)
        ax[1].plot(M.columns, M.median(0), "o-", color=col, lw=2, label=c.replace("kappa_", "κ, ") + " pairs")
        ax[1].fill_between(M.columns, M.quantile(.25, axis=0), M.quantile(.75, axis=0), color=col, alpha=.15, lw=0)
    ax[1].axhline(0, color="#999", lw=.7); ax[1].set_xscale("log"); ax[1].set_xticks([1, 5, 10, 20], ["1", "5", "10", "20"]); ax[1].set_xlabel("bin size (ms)")
    ax[1].set_ylabel("coupling scale κ before endpoint\n(median, IQR over sessions)"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("b  N2: kappa by timescale — UNVALIDATED (null test: biased upward)", loc="left", color="#a00", fontsize=8.5)
    ok = n3.D_supp_obs.notna() & n3.D_supp_pred.notna()
    ax[2].scatter(n3.D_supp_pred[ok], n3.D_supp_obs[ok], s=18, color=np.where(n3.index[ok] < 220000, "#2f6db3", "#c0392b"))
    lim = np.nanmax(np.abs(np.r_[n3.D_supp_pred[ok], n3.D_supp_obs[ok]])) * 1.1; ax[2].plot([-lim, lim], [-lim, lim], "--", color="#777", lw=1)
    ax[2].axhline(0, color="#ccc", lw=.6); ax[2].axvline(0, color="#ccc", lw=.6)
    ax[2].set_xlabel("predicted by fixed-J model (fields move)"); ax[2].set_ylabel("observed"); ax[2].set_title("c  N3: observed vs fixed-J prediction — UNVALIDATED (null FPR 12%)", loc="left", color="#a00", fontsize=8.5)
    fig.text(.01, -.03, "One dot per session (stim 0). Blue 2021, red 2022 (panel c). SI = coincidence / product of rates in 1-ms bins, minus 20-shuffle reach-shuffled value.", fontsize=8, color="#555")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"e14_nashef_reconciliation.{e}"), dpi=200, bbox_inches="tight")

def synth():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "synth_*.csv")))], ignore_index=True)
    assert d.session.nunique() == 4, d.session.nunique(); d.to_csv(os.path.join(TAB, "e14_synthetic_rows.csv"), index=False)
    m = 0.5 * float(d[d.scenario == "T0"].D_supp_obs.abs().median()); out = []
    for sc, g in d.groupby("scenario"):
        r = dict(scenario=sc, n=len(g), kappa_supp_median=float(g.kappa_supp.median()), kappa_supp_n_pos=int((g.kappa_supp > 0).sum()),
                 N3_resid_median=float(g.N3_resid.median()), N3_resid_n_pos=int((g.N3_resid > 0).sum()), D_supp_obs_median=float(g.D_supp_obs.median()),
                 class_agreement_median=float(g.class_agreement.median()), margin_m=m)
        if sc == "T0": r["pass"] = bool(g.kappa_supp.abs().median() < 0.15 and g.N3_resid.abs().median() < m)   # median of |.| (prereg)
        elif sc == "T1": r["pass"] = bool(abs(r["kappa_supp_median"] - 0.5) <= 0.25 and r["kappa_supp_n_pos"] >= 3 and r["N3_resid_n_pos"] >= 3)
        else: r["pass"] = "report (false-positive check)"
        out.append(r)
    T = pd.DataFrame(out); T.to_csv(os.path.join(TAB, "e14_synthetic_validation.csv"), index=False); print(T.to_string(index=False)); print(d.round(4).to_string(index=False))

def null():
    """Amendment A2: T0x bias and session-level false-positive rate for kappa_supp and the N3 residual."""
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "null_*.csv")))], ignore_index=True)
    d.to_csv(os.path.join(TAB, "e14_null_rows.csv"), index=False); rng = np.random.default_rng(SEED); out = []
    scale = float(d.D_supp_obs.abs().median())
    for c, bound in (("kappa_supp", 0.1), ("N3_resid", 0.1 * scale)):
        v = d[c].dropna().values; fp = 0
        for _ in range(10_000):
            s = rng.choice(v, 21, replace=True); fp += wil(s) < 0.05
        out.append(dict(stat=c, n_runs=len(v), n_sessions=int(d.loc[d[c].notna(), "session"].nunique()), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                        bias_bound=bound, fpr=fp / 10_000, pass_=bool(abs(v.mean()) <= bound and fp / 10_000 <= 0.07)))
    T = pd.DataFrame(out); T.to_csv(os.path.join(TAB, "e14_null_validation.csv"), index=False); print(T.to_string(index=False))

{"real": real, "synth": synth, "null": null}[MODE]()
