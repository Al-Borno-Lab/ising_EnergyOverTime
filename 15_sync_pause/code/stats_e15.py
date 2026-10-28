"""E15 statistics and figures (design: PREREGISTRATION_2026-10-27.md).
    python stats_e15.py real  RESULTS_DIR ASSETS_DIR     # e15_<session>.csv + e15_tc_<session>.npz
    python stats_e15.py synth SYNTH_DIR   ASSETS_DIR     # synth_<key>.csv"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MODE, SRC, ASSETS = sys.argv[1], sys.argv[2], sys.argv[3]
TAB, FIG = os.path.join(ASSETS, "tables"), os.path.join(ASSETS, "figures"); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
STATS = ["R_obs", "A_obs0", "A_10", "A_obs1", "L_obs0", "L_obs1"]
PRIMARY = ["A_obs0", "A_10", "A_obs1"]
EQ = np.log(1.25)
NBOOT, SEED = 10_000, 15
WINDOWS_PREREG = "{'B': (-100, -41), 'W': (-30, 10), 'P': (11, 50), 'C': (-200, -160)}"

def hl(x):
    """One-sample Hodges-Lehmann estimate: median of Walsh averages."""
    x = np.asarray(x, float); i, j = np.triu_indices(len(x)); return float(np.median((x[i] + x[j]) / 2))

def boot_ci(x, levels=(95, 90)):
    rng = np.random.default_rng(SEED); x = np.asarray(x, float)
    b = np.array([hl(x[rng.integers(0, len(x), len(x))]) for _ in range(NBOOT)])
    return {lv: tuple(np.percentile(b, [(100 - lv) / 2, 100 - (100 - lv) / 2])) for lv in levels}

def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x != 0) else float("nan")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out

def test_row(x, **lab):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; ci = boot_ci(x)
    return dict(**lab, n=len(x), median=float(np.median(x)), hl=hl(x), ci95_lo=ci[95][0], ci95_hi=ci[95][1],
                ci90_lo=ci[90][0], ci90_hi=ci[90][1], n_pos=int((x > 0).sum()), p=wil(x),
                equiv_125=bool(-EQ < ci[90][0] and ci[90][1] < EQ))

def load(pattern):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, pattern)))], ignore_index=True)
    assert (d.windows == WINDOWS_PREREG).all(), "windows differ from pre-registration"
    return d

def session_means(d):
    return d.groupby(["session", "event", "test_window"])[STATS].mean().reset_index()

def real():
    d = load("e15_[0-9]*.csv")
    assert d.session.nunique() == 21 and len(d.key.unique()) == 63, (d.session.nunique(), len(d.key.unique()))
    d.to_csv(os.path.join(TAB, "e15_conditions.csv"), index=False)
    S = session_means(d); S["cohort"] = np.where(S.session < 220000, "2021", "2022")
    # ---- primary family: pause, window W
    P = S[(S.event == "pause") & (S.test_window == "W")]
    prim = [test_row(P[s], family="primary", event="pause", test_window="W", stat=s, subset="all") for s in PRIMARY]
    assert all(np.isfinite(r["p"]) for r in prim), "non-finite primary p-value"
    for r, q in zip(prim, bh([r["p"] for r in prim])): r["q_bh"] = q
    desc = [test_row(P[s], family="descriptive", event="pause", test_window="W", stat=s, subset="all") for s in ("R_obs", "L_obs0", "L_obs1")]
    # ---- outcome (pre-registered rule)
    T = {r["stat"]: r for r in prim + desc}
    # sign convention (pre-registration: if pauses do not rise, "the same rules apply with signs read accordingly")
    sgn = 1.0 if T["R_obs"]["hl"] > 0 else -1.0
    R_up = sgn * T["R_obs"]["hl"] > 0                                   # rule (a): R_obs > 0 (direction of change)
    sig = lambda s: T[s]["q_bh"] < 0.05
    # precedence (amendment A2): (c) before (b) before (a)
    if sig("A_obs1"): outcome = "(c) beyond the fixed pairwise model"
    elif sig("A_obs0") and sgn * T["A_obs0"]["hl"] > 0 and sig("A_10") and sgn * T["A_10"]["hl"] > 0 and T["A_obs1"]["equiv_125"]:
        outcome = "(b) fixed coupling amplifies shared drive"
    elif R_up and not sig("A_obs0") and T["A_obs0"]["equiv_125"]: outcome = "(a) shared drive only"
    else: outcome = "inconclusive"
    # ---- secondary / sensitivity (not in the FDR family)
    sec = []
    n_by = d.groupby("session").N.min(); big = set(n_by[n_by >= 8].index)
    exact_only = set(d.groupby("session").method.agg(lambda m: (m == "exact").all()).loc[lambda x: x].index)
    for ev in ("pause", "burst", "pause05", "burst95"):
        for w in ("W", "P", "C"):
            x = S[(S.event == ev) & (S.test_window == w)]
            for s in STATS:
                if ev == "pause" and w == "W" and s in PRIMARY + ["R_obs", "L_obs0", "L_obs1"]: continue
                sec.append(test_row(x[s], family="secondary", event=ev, test_window=w, stat=s, subset="all"))
    for name, keep in [("cohort2021", lambda x: x.cohort == "2021"), ("cohort2022", lambda x: x.cohort == "2022"),
                       ("N>=8_all_blocks", lambda x: x.session.isin(big)), ("exact_only", lambda x: x.session.isin(exact_only))]:
        x = P[keep(P)]
        for s in PRIMARY + ["R_obs"]: sec.append(test_row(x[s], family="sensitivity", event="pause", test_window="W", stat=s, subset=name))
    tests = pd.DataFrame(prim + desc + sec); tests.to_csv(os.path.join(TAB, "e15_tests.csv"), index=False)
    # ---- calibration: predicted vs observed mean K
    c = d[(d.event == "pause")].groupby(["session", "test_window"])[["meanK_obs_X", "meanK_m0_X", "meanK_m1_X", "meanK_obs_B", "meanK_m0_B", "meanK_m1_B"]].mean().reset_index()
    cal = c.groupby("test_window").median(numeric_only=True).drop(columns="session"); cal.to_csv(os.path.join(TAB, "e15_calibration.csv"))
    deg = d[(d.e_obs_B > 0.999) & (d.e0_B > 0.999) & (d.e1_B > 0.999)]   # threshold at floor: event = every bin
    summ = dict(outcome=outcome, sign_of_R_obs=sgn, direction_ok=bool(R_up), degenerate_rows=int(len(deg)),
                degenerate_keys=sorted(set(deg.key + ":" + deg.event)), n_sessions=int(P.session.nunique()), n_conditions=int(len(d.key.unique())),
                gibbs_conditions=int((d.groupby("key").method.first() == "gibbs").sum()),
                primary={r["stat"]: {k: r[k] for k in ("hl", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "n_pos", "p", "q_bh", "equiv_125")} for r in prim},
                descriptive={r["stat"]: {k: r[k] for k in ("hl", "ci95_lo", "ci95_hi", "n_pos", "p")} for r in desc})
    json.dump(summ, open(os.path.join(TAB, "e15_summary.json"), "w"), indent=1, default=float)
    figure(d, S); print(json.dumps(summ, indent=1, default=float))

def figure(d, S):
    tc = {}
    for f in sorted(glob.glob(os.path.join(SRC, "e15_tc_*.npz"))):
        z = np.load(f)
        for n in z.files: tc[n] = z[n]
    keys = sorted(set(n.split("__")[0] for n in tc)); off = tc[f"{keys[0]}__offset"] * 10
    assert all(np.array_equal(tc[f"{k}__offset"] * 10, off) for k in keys)
    ses = sorted(set(k.split("_")[0] for k in keys))
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, ev in enumerate(("pause", "burst")):
        a = ax[row, 0]
        for s, lab, col, lw in (("o", "observed", "#111", 2), ("q0", "M0: rates only", "#e0861f", 1.6), ("q1", "M1: rates + fixed J", "#2f6db3", 1.6)):
            M = np.array([np.mean([tc[f"{k}__{ev}__{s}"] for k in keys if k.startswith(se)], 0) for se in ses])
            a.plot(off, np.median(M, 0), color=col, lw=lw, label=lab)
        for w, (lo, hi), c in (("W", (-300, 100), "#f2d3d3"), ("B", (-1000, -410), "#e8e8e8"), ("C", (-2000, -1600), "#dfe9f5")):
            a.axvspan(lo, hi, color=c, lw=0, zorder=0)
        a.axvline(0, color="k", ls=":", lw=.7); a.set_xlabel("time from peak velocity (ms)"); a.set_ylabel(f"P({ev}) per 10-ms bin, held out")
        a.set_title(f"{'ad'[row]}  {ev}: K {'<=' if ev == 'pause' else '>='} baseline {'10th' if ev == 'pause' else '90th'} percentile", loc="left")
        if row == 0: a.legend(frameon=False, fontsize=8)
        for j, w in enumerate(("W", "C")):
            a = ax[row, j + 1]; x = S[(S.event == ev) & (S.test_window == w)]
            for i, s in enumerate(("R_obs", "A_obs0", "A_10", "A_obs1")):
                v = x[s].values; a.scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=14, color="#1f4e99", alpha=.7)
                a.plot([i - .25, i + .25], [np.median(v)] * 2, color="k", lw=2)
            a.axhline(0, color="#999", lw=.7); a.axhspan(-EQ, EQ, color="#eee", lw=0, zorder=0)
            a.set_xticks(range(4), ["event rate\nchange", "data vs\nrates-only", "coupling\nprediction", "data vs\nM1"])
            a.set_ylabel("log ratio, window minus baseline"); a.set_title(f"{'be'[row] if j == 0 else 'cf'[row]}  {'reach window (-300..+100 ms)' if w == 'W' else 'rest control (-2000..-1600 ms)'}", loc="left")
    fig.text(.01, -.02, "One dot per session (mean of 3 blocks), n = 21. Held-out 5-fold CV. Grey band = equivalence margin ±log(1.25). Rest control should be ~0.", fontsize=8, color="#555")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"e15_pause_timecourse.{e}"), dpi=200, bbox_inches="tight")

def synth():
    d = load("synth_*.csv"); d.to_csv(os.path.join(TAB, "e15_synthetic_conditions.csv"), index=False)
    x = d[(d.event == "pause") & (d.test_window == "W")]; rows = []
    for sc, g in x.groupby("scenario"):
        assert len(g) == 6, (sc, len(g))
        r = dict(scenario=sc, n=len(g), **{f"median_{s}": float(g[s].median()) for s in ("R_obs", "A_obs0", "A_10", "A_obs1")},
                 **{f"median_abs_{s}": float(g[s].abs().median()) for s in ("A_obs0", "A_10", "A_obs1")}, n_A_obs1_pos=int((g.A_obs1 > 0).sum()))
        if sc == "S0": r["pass"] = all(r[f"median_abs_{s}"] < 0.1 for s in ("A_obs0", "A_10", "A_obs1"))
        elif sc == "S1": r["pass"] = r["median_abs_A_obs1"] < 0.1
        else: r["pass"] = r["n_A_obs1_pos"] >= 5
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(TAB, "e15_synthetic_validation.csv"), index=False); print(T.to_string(index=False))

{"real": real, "synth": synth}[MODE]()
