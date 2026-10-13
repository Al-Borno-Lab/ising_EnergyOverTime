"""E12 statistics: session-level T1/T2/T3 per signal, pre-registered comparisons (primary BH family, secondary family),
U1/U2 with bootstrap CIs + Hodges-Lehmann, cohort and mouse breakdowns. Reads rows.csv from compute_signals.py."""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

MOUSE = {**{s: "fChR2" for s in (210421, 210422, 210423, 210425, 210514)},
         **{s: "fChR5" for s in (210511, 210512, 210515)}, **{s: "fChR4" for s in (210606, 210608, 210614)},
         **{s: "cChR1" for s in (210619, 210620, 210622, 210623)},
         **{s: "DJC002" for s in (220515, 220517, 220519)}, **{s: "DJC000" for s in (220516, 220518, 220520)}}
TOL = dict(T1=0.10, T2=15.0)      # equivalence margins: d1 fraction; T2 in ms (1 bin = 10 ms)
PRIMARY = [("T1", "FR"), ("T1", "dFR"), ("T1", "FT0"), ("T1", "POLb"), ("T1", "TRAJ"),
           ("T2", "FR"), ("T2", "dFR"), ("T2", "DEC"), ("T2", "FT0"), ("T2", "POLb"), ("T2", "TRAJ"), ("T3", "FR")]
rng = np.random.RandomState(12)


def session_metrics(rows):
    rows = rows.copy()
    rows["session"] = rows.cond.str.split("_").str[0].astype(int)
    real = rows[rows["shift"] == 0].set_index(["cond", "signal"])
    ctrl = rows[rows["shift"] != 0].groupby(["cond", "signal"]).agg(hit=("hit", "mean"), depth=("depth", "mean"), dev=("dev", "mean"), n=("hit", "size"))
    c = real.join(ctrl, rsuffix="_c", how="inner")
    c["ratio"] = c["depth"] / c["depth_c"]
    c = c.reset_index(); c["session"] = c.cond.str.split("_").str[0].astype(int)
    g = c.groupby(["session", "signal"])
    s = g.agg(hit=("hit", "mean"), hit_c=("hit_c", "mean"), dev=("dev", "median"), dev_c=("dev_c", "median"),
              ratio=("ratio", "median"), depth=("depth", "median"), nconds=("cond", "size"), rel_pa=("rel_pa", "median")).reset_index()
    s["d1"] = s.hit - s.hit_c; s["dev_ms"] = s.dev * 10; s["dev_c_ms"] = s.dev_c * 10
    s["logratio"] = np.log(s.ratio)
    s["cohort"] = np.where(s.session < 220000, 2021, 2022); s["mouse"] = s.session.map(MOUSE)
    return s


METRIC = dict(T1="d1", T2="dev_ms", T3="logratio")


def wil(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0): return np.nan, 1.0
    r = stats.wilcoxon(x, zero_method="wilcox")
    return r.statistic, r.pvalue


def hl(x):
    x = np.asarray(x); w = (x[:, None] + x[None]) / 2
    return float(np.median(w[np.triu_indices(len(x))]))


def boot_ci(x, nb=10000):
    x = np.asarray(x, float); n = len(x)
    m = np.array([np.mean(x[rng.randint(0, n, n)]) for _ in range(nb)])
    return np.percentile(m, [2.5, 97.5])


def pivot(s, test, sig):
    return s[s.signal == sig].set_index("session")[METRIC[test]]


def compare(s, test, a, b):
    """A minus B on the session metric. For T2 (lower = better) a negative difference favours A."""
    d = (pivot(s, test, a) - pivot(s, test, b)).dropna()
    st, p = wil(d.values)
    lo, hi = boot_ci(d.values)
    tol = TOL.get(test)
    verdict = ("tie" if (tol is not None and -tol <= lo and hi <= tol) else
               ("A_better" if ((hi < 0 and test == "T2") or (lo > 0 and test != "T2")) else
                ("B_better" if ((lo > 0 and test == "T2") or (hi < 0 and test != "T2")) else "unresolved")))
    out = dict(test=test, A=a, B=b, n=len(d), mean_diff=d.mean(), hl_diff=hl(d.values), ci_lo=lo, ci_hi=hi,
               median_A=pivot(s, test, a).median(), median_B=pivot(s, test, b).median(), p=p, verdict_ci=verdict)
    for coh in (2021, 2022):
        dc = d[d.index.isin(s[s.cohort == coh].session)]
        out[f"p_{coh}"] = wil(dc.values)[1]; out[f"mean_{coh}"] = dc.mean(); out[f"n_{coh}"] = len(dc)
    # mouse random intercept
    df = pd.DataFrame(dict(d=d.values, mouse=[MOUSE[i] for i in d.index]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("d ~ 1", df, groups=df.mouse).fit(reml=True)
        out.update(mm_est=float(m.params["Intercept"]), mm_lo=float(m.conf_int().loc["Intercept", 0]), mm_hi=float(m.conf_int().loc["Intercept", 1]))
    except Exception:
        out.update(mm_est=np.nan, mm_lo=np.nan, mm_hi=np.nan)
    return out


def against_control(s, sig):
    out = []
    for test, col, null in (("T1", "d1", 0.0), ("T3", "logratio", 0.0)):
        x = s[s.signal == sig][col].dropna().values - null
        st, p = wil(x)
        out.append(dict(signal=sig, test=test, n=len(x), median=float(np.median(x)), mean=float(np.mean(x)), p=p))
    return out


def run(rows_csv, out, anchor="FT"):
    os.makedirs(out, exist_ok=True)
    rows = pd.read_csv(rows_csv)
    s = session_metrics(rows); s.to_csv(f"{out}/session_metrics.csv", index=False)
    # per-signal summary
    summ = s.groupby("signal").agg(sessions=("session", "nunique"), hit_real=("hit", "mean"), hit_ctrl=("hit_c", "mean"),
                                   d1_mean=("d1", "mean"), dev_ms_median=("dev_ms", "median"), dev_ctrl_ms_median=("dev_c_ms", "median"),
                                   depth_ratio_median=("ratio", "median"), rel_pa_ms_median=("rel_pa", lambda v: 10 * v.median())).reset_index()
    for coh in (2021, 2022):
        h = s[s.cohort == coh].groupby("signal").agg(**{f"hit_{coh}": ("hit", "mean"), f"hitc_{coh}": ("hit_c", "mean"), f"dev_ms_{coh}": ("dev_ms", "median")}).reset_index()
        summ = summ.merge(h, on="signal", how="left")
    summ.to_csv(f"{out}/signal_summary.csv", index=False)
    # primary family
    prim = pd.DataFrame([compare(s, t, anchor, b) for t, b in PRIMARY if b in set(s.signal)])
    prim["p_bh"] = multipletests(prim.p.fillna(1), method="fdr_bh")[1]
    prim.to_csv(f"{out}/primary_family.csv", index=False)
    # secondary family
    sec = pd.DataFrame([r for sig in sorted(set(s.signal)) for r in against_control(s, sig)])
    sec["p_bh"] = multipletests(sec.p.fillna(1), method="fdr_bh")[1]
    sec.to_csv(f"{out}/secondary_vs_control.csv", index=False)
    # extra descriptive contrasts (not in the family)
    extra = pd.DataFrame([compare(s, t, anchor, b) for t, b in (("T1", "FR_abs"), ("T2", "FR_abs"), ("T3", "dFR"), ("T3", "FT0"), ("T3", "POLb"), ("T3", "TRAJ"), ("T1", "FTu"), ("T2", "FTu"), ("T3", "FTu"), ("T1", "TRAJd"), ("T2", "TRAJd")) if b in set(s.signal)])
    extra.to_csv(f"{out}/descriptive_contrasts.csv", index=False)
    return s, summ, prim, sec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); run(a.rows, a.out)
