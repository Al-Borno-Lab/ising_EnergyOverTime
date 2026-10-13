"""Statistics for E11 as pre-registered (PREREGISTRATION_2026-10-12.md section 6-7).
usage: python analyze_stimfit.py <results_dir> <assets_tables_dir> [--selftest]
Reads <results_dir>/tasks/*.json. Writes session_table.csv and stats_summary.json (+ variant tables) into <assets_tables_dir>.
Unit = session. Wilcoxon signed-rank (two-sided), Hodges-Lehmann estimate with exact 90% CI, BH over {S1, S2}."""
import sys, os, glob, json, csv, itertools
import numpy as np
from scipy import stats

COHORT = {"2021": lambda d: d.startswith("21"), "2022": lambda d: d.startswith("22")}
MOUSE = {"210421": "fChR2", "210422": "fChR2", "210423": "fChR2", "210425": "fChR2", "210514": "fChR2",
         "210511": "fChR5", "210512": "fChR5", "210515": "fChR5", "210606": "fChR4", "210608": "fChR4", "210614": "fChR4",
         "210619": "cChR1", "210620": "cChR1", "210622": "cChR1", "210623": "cChR1",
         "220515": "DJC002", "220517": "DJC002", "220519": "DJC002", "220516": "DJC000", "220518": "DJC000", "220520": "DJC000"}
EQ_BOUND = 0.3


def hl_ci(x, level=0.90):
    """Hodges-Lehmann median of Walsh averages with distribution-free CI at `level` (exact signed-rank)."""
    x = np.asarray(x, float); n = len(x)
    w = np.array([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)]); w.sort()
    M = len(w)
    counts = np.zeros(M + 1); counts[0] = 1
    for k in range(1, n + 1):
        new = counts.copy(); new[k:] += counts[:M + 1 - k]; counts = new
    cdf = np.cumsum(counts) / 2.0 ** n
    alpha = 1 - level
    ok = np.nonzero(cdf <= alpha / 2)[0]
    c = int(ok.max()) if len(ok) else -1                                  # largest c with P(W <= c) <= alpha/2
    lo, hi = (w[c], w[M - 1 - c]) if c >= 0 else (w[0], w[-1])           # [w_(c+1), w_(M-c)] in 1-based order
    return float(np.median(w)), float(lo), float(hi)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); q = np.empty(m)
    run = 1.0
    for rank, i in zip(range(m, 0, -1), o[::-1]):
        run = min(run, p[i] * m / rank); q[i] = run
    return q


def load(res_dir):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(res_dir, "tasks", "*.json"))]
    by = {}
    for r in rows: by.setdefault((r["session"], r["subset"]), {})[r["kind"]] = r
    return by


def session_values(by, kind="primary", corrected=True):
    """Per session: median over subsets of Dkappa (null-corrected if requested)."""
    out = {}
    for (sess, sub), d in sorted(by.items()):
        if kind not in d or not d[kind]["converged"]: continue
        kp, kc = d[kind]["kappa"][:2] if len(d[kind]["kappa"]) >= 2 else (np.nan, np.nan)
        if d[kind]["groups"][:2] == ["pulse_early", "pulse_late"]: continue
        sh = [v for k, v in d.items() if k.startswith("shuf_") and v["converged"]]
        if corrected:
            if len(sh) < 10: continue
            kp -= np.mean([s["kappa"][0] for s in sh]); kc -= np.mean([s["kappa"][1] for s in sh])
        out.setdefault(sess, []).append(kp - kc)
    return {s: float(np.median(v)) for s, v in out.items()}


def test(vals, label, kappa_scale=True):
    x = np.array(list(vals.values())); n = len(x)
    if n < 6: return dict(label=label, n=n, note="too few sessions")
    w = stats.wilcoxon(x, alternative="two-sided")
    hl, lo, hi = hl_ci(x)
    return dict(label=label, n=n, median=float(np.median(x)), hl=hl, ci90=[lo, hi], p=float(w.pvalue),
                equivalent=(bool(lo > -EQ_BOUND and hi < EQ_BOUND) if kappa_scale else None))


def main(res_dir, out_dir):
    by = load(res_dir); os.makedirs(out_dir, exist_ok=True); S = {}
    prim = session_values(by, "primary", True)
    S["S1"] = test(prim, "S1 null-corrected dkappa")
    S["S1_raw"] = test(session_values(by, "primary", False), "S1 raw dkappa")
    for k, name in (("round", "round"), ("shift1", "floor+1"), ("fixedctrl", "fixed control 401-406"), ("allreach", "all reaches")):
        S["sens_" + k] = test(session_values(by, k, False), "sensitivity " + name + " (raw dkappa)")
    S["sens_without_seen_sessions"] = test({s: v for s, v in prim.items() if s not in ("210511", "220519")}, "S1 without 210511 and 220519 (seen before the full run)")
    S["sens_N5"] = test({s: v for s, v in prim.items() if s not in ("210622", "210623")}, "S1 without N<5 sessions")
    for c, f in COHORT.items(): S["cohort_" + c] = test({s: v for s, v in prim.items() if f(s)}, "S1 cohort " + c)
    # S2: dI/N and dchi/N medians over subsets
    for key in ("dI_per_N", "dchiJ_per_N"):
        v = {}
        for (sess, sub), d in by.items():
            if "primary" in d and d["primary"]["converged"] and key in d["primary"]: v.setdefault(sess, []).append(d["primary"][key])
        S["S2_" + key] = test({s: float(np.median(a)) for s, a in v.items()}, "S2 " + key, kappa_scale=False)
    # S3
    for a, b, name in (("pulse_early", "pulse_late", "early-late"), ("washout", "control", "washout-control")):
        v = {}
        for (sess, sub), d in by.items():
            if "s3" in d and d["s3"]["converged"]:
                g = d["s3"]["groups"]; k = d["s3"]["kappa"]; v.setdefault(sess, []).append(k[g.index(a)] - k[g.index(b)])
        S["S3_" + name] = test({s: float(np.median(x)) for s, x in v.items()}, "S3 " + name)
    # BH over {S1, S2 (dI/N)}
    fam = [S["S1"], S["S2_dI_per_N"]]
    if all("p" in f for f in fam):
        q = bh([f["p"] for f in fam]); fam[0]["p_fdr"], fam[1]["p_fdr"] = float(q[0]), float(q[1])
    # mouse random-effect model
    try:
        import pandas as pd, statsmodels.formula.api as smf
        df = pd.DataFrame(dict(session=list(prim), y=list(prim.values()))); df["mouse"] = df.session.map(MOUSE)
        m = smf.mixedlm("y ~ 1", df, groups=df["mouse"]).fit(reml=True)
        S["mixed_mouse"] = dict(intercept=float(m.params["Intercept"]), ci95=[float(v) for v in m.conf_int().loc["Intercept"]], n_mice=int(df.mouse.nunique()))
    except Exception as e:
        S["mixed_mouse"] = dict(error=str(e))
    with open(os.path.join(out_dir, "session_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["session", "mouse", "cohort", "dkappa_null_corrected"])
        for s, v in sorted(prim.items()): w.writerow([s, MOUSE.get(s, "?"), "2021" if s.startswith("21") else "2022", v])
    json.dump(S, open(os.path.join(out_dir, "stats_summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


def selftest():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.1, 21); hl, lo, hi = hl_ci(x)
    print("HL %.3f CI90 [%.3f, %.3f] median %.3f" % (hl, lo, hi, np.median(x)))
    # exact coverage check: CI at 90% should cover 0 about 90% of the time under symmetric noise
    cov = np.mean([(lambda c: c[1] <= 0 <= c[2])(hl_ci(rng.normal(0, 1, 21))) for _ in range(400)])
    print("coverage of true median (target 0.90): %.3f" % cov)
    print("BH:", bh([0.01, 0.04]))
    assert 0.85 <= cov <= 0.96


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main(sys.argv[1], sys.argv[2])
