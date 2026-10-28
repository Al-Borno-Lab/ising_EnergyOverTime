"""Statistics for E11 as pre-registered (PREREGISTRATION_2026-10-12.md sections 3, 6, 7 and amendments A3, A4, A5).
usage: python analyze_stimfit.py <results_dir> <assets_tables_dir> [--exact <exact_results_dir>] | --selftest
Reads <results_dir>/tasks/*.json. Writes session_table.csv and stats_summary.json into <assets_tables_dir>.
PRIMARY statistic (A4) = RAW kappa_pulse - kappa_control, session median over subsets. The shuffle null is a labelled DIAGNOSTIC only.
Unit = session. Wilcoxon signed-rank (two-sided), Hodges-Lehmann estimate with exact 90% CI, TOST-equivalent CI rule, BH over {S1, S2}.
Completeness (section 3 + A5): every expected design/kind must be present (else status "incomplete", no verdict); a session needs >= 5 converged
finite subsets (N > 16) or its one converged finite fit (N <= 16) for each kind (else it FAILS that kind and is left out, status "n_reduced")."""
import sys, os, glob, json, csv, warnings
import numpy as np
from scipy import stats

COHORT = {"2021": lambda d: d.startswith("21"), "2022": lambda d: d.startswith("22")}
MOUSE = {"210421": "fChR2", "210422": "fChR2", "210423": "fChR2", "210425": "fChR2", "210514": "fChR2",
         "210511": "fChR5", "210512": "fChR5", "210515": "fChR5", "210606": "fChR4", "210608": "fChR4", "210614": "fChR4",
         "210619": "cChR1", "210620": "cChR1", "210622": "cChR1", "210623": "cChR1",
         "220515": "DJC002", "220517": "DJC002", "220519": "DJC002", "220516": "DJC000", "220518": "DJC000", "220520": "DJC000"}
NEURONS = {"210421": 15, "210422": 11, "210423": 5, "210425": 5, "210511": 12, "210512": 8, "210514": 11, "210515": 18, "210606": 30, "210608": 21,
           "210614": 25, "210619": 31, "210620": 32, "210622": 3, "210623": 4, "220515": 19, "220516": 19, "220517": 24, "220518": 27, "220519": 17, "220520": 27}
SEEN = ("210511", "220519")                      # fitted on real data before run 1 (disclosure, A2)
CORE_KINDS = ("primary", "round", "shift1", "fixedctrl", "allreach", "s3"); NSHUF = 10
EQ_BOUND = 0.3; RANGE = [300, 560]


def n_subsets(s): return 1 if NEURONS[s] <= 16 else 10
def min_ok(s): return 1 if NEURONS[s] <= 16 else 5


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
    bad = [(r["session"], r["kind"], r.get("t_range")) for r in rows if r.get("t_range") != RANGE]
    assert not bad, "fits outside the pre-registered range 300-559: %s" % bad[:3]                 # A3
    bad = [(r["session"], r["kind"]) for r in rows if r.get("shuffle") != "aligned (A3)" or "t_k = bin start" not in str(r.get("onset_convention"))]
    assert not bad, "rows not produced by the A3 pipeline: %s" % bad[:3]
    by = {}
    for r in rows: by.setdefault((r["session"], r["subset"]), {})[r["kind"]] = r
    return by


def usable(r):
    """Converged and every kappa (and, for primary/exact rows, the S2 quantities) finite; non-finite counts as NOT converged (A5)."""
    if not r["converged"] or not np.all(np.isfinite(r["kappa"])): return False
    return all(np.isfinite(r[k]) for k in ("dI_per_N", "dchiJ_per_N") if k in r)


def check_design(by):
    """Expected design: every session, n_subsets(s) subsets, all core kinds and NSHUF shuffles. Returns missing (files absent), nonusable and unexpected sessions."""
    missing, nonusable = [], []
    for s in MOUSE:
        for sub in range(n_subsets(s)):
            d = by.get((s, sub), {})
            for k in CORE_KINDS + tuple("shuf_%d" % j for j in range(NSHUF)):
                if k not in d: missing.append((s, sub, k))
                elif not usable(d[k]): nonusable.append((s, sub, k))
    extra = sorted({s for (s, _) in by} - set(MOUSE))
    return missing, nonusable, extra


def session_values(by, kind="primary", corrected=False, min_shuf=7):
    """Per session: median over subsets of the per-fit dkappa = kappa_pulse - kappa_control, using usable fits only.
    A session with fewer than min_ok(s) usable subsets FAILS this kind (listed). corrected=True is the DIAGNOSTIC null-corrected value."""
    out, failed, below = {}, [], set()
    for s in MOUSE:
        vals = []
        for sub in range(n_subsets(s)):
            d = by.get((s, sub), {})
            if kind not in d or not usable(d[kind]): continue
            kp, kc = d[kind]["kappa"][:2]
            if min(d[kind]["kappa"]) < -1: below.add(s)
            if corrected:
                sh = [v for k, v in d.items() if k.startswith("shuf_") and usable(v)]
                if len(sh) < min_shuf: continue
                kp = kp - np.mean([x["kappa"][0] for x in sh]); kc = kc - np.mean([x["kappa"][1] for x in sh])
            vals.append(kp - kc)
        if len(vals) >= min_ok(s): out[s] = float(np.median(vals))
        else: failed.append(s)
    return out, failed, sorted(below)


def outcome(hl, ci, p_rule):
    """Pre-registered decision (PREREG section 6, A4, A5). a: p < 0.05 and |HL| >= 0.1; b: 90% CI inside (-0.3, 0.3); c: otherwise.
    A5: if both a and b hold, report (a) and state that the change is bounded within +-0.3 (bounded=True)."""
    a = bool(p_rule < 0.05 and abs(hl) >= 0.1); b = bool(ci[0] > -EQ_BOUND and ci[1] < EQ_BOUND)
    return ("a" if a else "b" if b else "c"), bool(a and b), (("decrease" if hl < 0 else "increase") if a else None)


def test(vals, label, n_expected, status_global, kappa_scale=True, failed=(), below=()):
    x = np.array(list(vals.values())); n = len(x)
    base = dict(label=label, n=n, n_expected=n_expected, sessions_failed_kind=list(failed), sessions_with_kappa_below_minus1=list(below))
    if n < 6: return dict(base, status="incomplete", note="too few sessions", outcome=None)
    w = stats.wilcoxon(x, alternative="two-sided"); hl, lo, hi = hl_ci(x)
    status = "incomplete" if status_global == "incomplete" else ("complete" if n == n_expected else "n_reduced")
    return dict(base, status=status, median=float(np.median(x)), hl=hl, ci90=[lo, hi], p=float(w.pvalue),
                equivalent=(bool(lo > -EQ_BOUND and hi < EQ_BOUND) if kappa_scale else None))


def verdict(t, use_fdr=False):
    """Adds the pre-registered outcome to a kappa-scale test. S1 uses p_FDR; sensitivity rows use the raw p and are DESCRIPTIVE. Withheld if incomplete."""
    if t.get("status") == "incomplete" or "hl" not in t or (use_fdr and "p_fdr" not in t):
        t["outcome"] = None; return t
    t["outcome"], t["bounded_within_0.3"], t["direction"] = outcome(t["hl"], t["ci90"], t["p_fdr"] if use_fdr else t["p"])
    t["outcome_basis"] = "p_FDR (pre-registered)" if use_fdr else "raw p (descriptive)"
    return t


def mixed_model(prim):
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame(dict(session=list(prim), y=list(prim.values()))); df["mouse"] = df.session.map(MOUSE)
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        try:
            m = smf.mixedlm("y ~ 1", df, groups=df["mouse"]).fit(reml=True)
            re_var = float(np.asarray(m.cov_re)[0, 0]); ci = [float(v) for v in m.conf_int().loc["Intercept"]]
            msgs = sorted({str(w.message)[:120] for w in W})
            unreliable = bool((not m.converged) or re_var < 1e-8 or msgs)
            return dict(intercept=float(m.params["Intercept"]), ci95=ci, ci_level="95% (Wald, REML)", n_mice=int(df.mouse.nunique()), converged=bool(m.converged),
                        random_effect_variance=re_var, warnings=msgs, ci_reliable=not unreliable)
        except Exception as e:
            return dict(error=str(e), warnings=sorted({str(w.message)[:120] for w in W}), ci_reliable=False)


def below_m1_counts(by):
    """kappa < -1 flips the sign of J_eff. Counts over usable fits only, for every kind including s3 (all four kappa) and the shuffles."""
    out = {}
    for kind in CORE_KINDS + ("shuffles",):
        n = tot = 0; sess = set()
        for (s, sub), d in by.items():
            for k, r in d.items():
                if (k == kind) or (kind == "shuffles" and k.startswith("shuf_")):
                    if usable(r):
                        tot += 1
                        if min(r["kappa"]) < -1: n += 1; sess.add(s)
        out[kind] = dict(fits_below_minus1=n, usable_fits=tot, sessions=sorted(sess))
    return out


def exact_check(by, exact_dir):
    fs = glob.glob(os.path.join(exact_dir, "tasks", "*_exact.json"))
    if not fs: return dict(status="missing")
    e = json.load(open(fs[0])); s = e["session"]
    sub = [d["primary"]["kappa"][:2] for (ss, _), d in by.items() if ss == s and "primary" in d and usable(d["primary"])]
    if not sub: return dict(status="no subset fits", session=s)
    med = np.median(np.array(sub), axis=0)
    return dict(status="ok", session=s, N=e["N"], exact_kappa_pulse_control=e["kappa"][:2], subset_median_kappa_pulse_control=[float(v) for v in med],
                exact_dkappa=e["kappa"][0] - e["kappa"][1], subset_median_dkappa=float(med[0] - med[1]), t_range=e["t_range"], converged=e["converged"], n_subsets=len(sub))


def main(res_dir, out_dir, exact_dir=None, quiet=False):
    by = load(res_dir); os.makedirs(out_dir, exist_ok=True); S = {}
    missing, nonusable, extra = check_design(by)
    status_global = "incomplete" if (missing or extra) else "complete"
    S["integrity"] = dict(t_range="all rows [300, 560], shuffle aligned (A3)", sessions_found=sorted({s for s, _ in by}), fits=sum(len(d) for d in by.values()),
                          missing_files=len(missing), missing_examples=missing[:10], unexpected_sessions=extra, nonusable_fits=len(nonusable), nonusable_examples=nonusable[:10],
                          nonconverged=sum(1 for d in by.values() for v in d.values() if not v["converged"]),
                          nonfinite=sum(1 for d in by.values() for v in d.values() if v["converged"] and not usable(v)), reran=sum(1 for d in by.values() for v in d.values() if v.get("reran")),
                          status=status_global)
    prim, f_prim, b_prim = session_values(by, "primary")
    S["S1"] = test(prim, "S1 RAW dkappa = kappa_pulse - kappa_control (primary, A4)", 21, status_global, failed=f_prim, below=b_prim)
    diag, f_diag, _ = session_values(by, "primary", True)
    S["diag_null_corrected"] = test(diag, "DIAGNOSTIC null-corrected dkappa (aligned shuffle, 10/subset); not a test of record", 21, status_global, failed=f_diag)
    for key in ("p", "equivalent"): S["diag_null_corrected"].pop(key, None)                    # cannot be quoted as a result
    for k, name in (("round", "round"), ("shift1", "floor+1"), ("fixedctrl", "fixed control 401-406"), ("allreach", "all reaches")):
        v, f, b = session_values(by, k)
        S["sens_" + k] = test(v, "sensitivity " + name + " (raw dkappa)", 21, status_global, failed=f, below=b)
    n_lt5 = sorted(s for s in MOUSE if NEURONS[s] < 5)
    assert n_lt5 == ["210622", "210623"], n_lt5
    S["sens_without_seen_sessions"] = test({s: v for s, v in prim.items() if s not in SEEN}, "S1 without 210511 and 220519 (seen before run 1)", 19, status_global)
    S["sens_N5"] = test({s: v for s, v in prim.items() if s not in n_lt5}, "S1 without N<5 sessions", 19, status_global)
    for c, f in COHORT.items(): S["cohort_" + c] = test({s: v for s, v in prim.items() if f(s)}, "S1 cohort " + c, 15 if c == "2021" else 6, status_global)
    for key in ("dI_per_N", "dchiJ_per_N"):                                                  # S2
        v, fl = {}, []
        for s in MOUSE:
            a = [d["primary"][key] for sub in range(n_subsets(s)) for d in [by.get((s, sub), {})] if "primary" in d and usable(d["primary"])]
            if len(a) >= min_ok(s): v[s] = float(np.median(a))
            else: fl.append(s)
        S["S2_" + key] = test(v, "S2 " + key, 21, status_global, kappa_scale=False, failed=fl)
    for a, b, name in (("pulse_early", "pulse_late", "early-late"), ("washout", "control", "washout-control")):   # S3 (exploratory)
        v, fl, bl = {}, [], set()
        for s in MOUSE:
            vals = []
            for sub in range(n_subsets(s)):
                d = by.get((s, sub), {})
                if "s3" in d and usable(d["s3"]):
                    g = d["s3"]["groups"]; k = d["s3"]["kappa"]; vals.append(k[g.index(a)] - k[g.index(b)])
                    if min(k) < -1: bl.add(s)
            if len(vals) >= min_ok(s): v[s] = float(np.median(vals))
            else: fl.append(s)
        S["S3_" + name] = test(v, "S3 " + name + " (exploratory)", 21, status_global, failed=fl, below=sorted(bl))
    fam = [S["S1"], S["S2_dI_per_N"]]                                                        # BH over {S1, S2 dI/N}
    if all("p" in f for f in fam):
        q = bh([f["p"] for f in fam]); fam[0]["p_fdr"], fam[1]["p_fdr"] = float(q[0]), float(q[1])
    S["S1"] = verdict(S["S1"], use_fdr=True)
    for k in [k for k in S if k.startswith(("sens_", "cohort_"))]: S[k] = verdict(S[k])
    S["mixed_mouse"] = mixed_model(prim) if len(prim) >= 6 else dict(error="too few sessions")
    S["kappa_below_minus1"] = below_m1_counts(by)
    if exact_dir: S["exactness_check"] = exact_check(by, exact_dir)
    nd = {}
    for (sess, sub), d in by.items():
        sh = [v for k, v in d.items() if k.startswith("shuf_") and usable(v)]
        if len(sh) > 1: nd.setdefault(sess, []).append((np.std([x["kappa"][0] for x in sh], ddof=1), np.std([x["kappa"][1] for x in sh], ddof=1), np.mean([x["kappa"][0] - x["kappa"][1] for x in sh])))
    with open(os.path.join(out_dir, "session_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["session", "mouse", "cohort", "dkappa_raw_primary", "dkappa_null_corrected_diagnostic", "null_sd_kappa_pulse", "null_sd_kappa_control", "null_mean_diff"])
        for s, v in sorted(prim.items()):
            a = np.array(nd.get(s, [(np.nan,) * 3]))
            w.writerow([s, MOUSE.get(s, "?"), "2021" if s.startswith("21") else "2022", v, diag.get(s, ""), float(np.nanmedian(a[:, 0])), float(np.nanmedian(a[:, 1])), float(np.nanmedian(a[:, 2]))])
    json.dump(S, open(os.path.join(out_dir, "stats_summary.json"), "w"), indent=1, default=str)
    if not quiet: print(json.dumps(S, indent=1, default=str))
    return S


def selftest():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.1, 21); hl, lo, hi = hl_ci(x)
    print("HL %.3f CI90 [%.3f, %.3f] median %.3f" % (hl, lo, hi, np.median(x)))
    cov = np.mean([(lambda c: c[1] <= 0 <= c[2])(hl_ci(rng.normal(0, 1, 21))) for _ in range(400)])
    print("coverage of true median (target 0.90): %.3f" % cov); print("BH:", bh([0.01, 0.04]))
    assert 0.85 <= cov <= 0.96


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else:
        ex = sys.argv[sys.argv.index("--exact") + 1] if "--exact" in sys.argv else None
        main(sys.argv[1], sys.argv[2], ex)
