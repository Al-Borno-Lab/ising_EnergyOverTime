"""E16 statistics and figure (design: PREREGISTRATION_2026-11-03.md).
    python stats_e16.py synth SYNTH_DIR ASSETS_DIR      python stats_e16.py real REAL_DIR ASSETS_DIR"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
MODE, SRC, ASSETS = sys.argv[1], sys.argv[2], sys.argv[3]
TAB, FIG = os.path.join(ASSETS, "tables"), os.path.join(ASSETS, "figures"); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
NBOOT, SEED = 10_000, 16

def hl(x):
    x = np.asarray(x, float); i, j = np.triu_indices(len(x)); return float(np.median((x[i] + x[j]) / 2))
def boot(x):
    rng = np.random.default_rng(SEED); x = np.asarray(x, float); b = np.array([hl(x[rng.integers(0, len(x), len(x))]) for _ in range(NBOOT)])
    return tuple(np.percentile(b, [2.5, 97.5]))
def wil(x):
    x = np.asarray(x, float); return float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x != 0) else float("nan")
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out

def session_D(d, basis, col, subset=None):
    x = d[d.basis == basis] if subset is None else d[(d.basis == basis) & subset(d)]
    reach = x[x.window == "reach"].groupby("session")[col].mean()
    rest = x[x.window.str.startswith("rest")].groupby("session")[col].mean()
    return (reach - rest).dropna(), reach, rest

def row(D, **lab):
    lo, hi = boot(D.values) if len(D) > 1 else (np.nan, np.nan)
    return dict(**lab, n=len(D), hl=hl(D.values), ci95_lo=lo, ci95_hi=hi, n_pos=int((D > 0).sum()), n_neg=int((D < 0).sum()), p=wil(D.values))

def real():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "e16_[0-9]*.csv")))], ignore_index=True)
    assert d.session.nunique() == 21 and d.key.nunique() == 63, (d.session.nunique(), d.key.nunique())
    assert (d.search == "(-200, 100)").all() and (d.hit_ms == 30).all() and (d.smooth_ms == 5).all()
    d.to_csv(os.path.join(TAB, "e16_rows.csv"), index=False)
    prim = [row(session_D(d, "U0", c)[0], family="primary", basis="U0", stat=c) for c in ("hit_peak", "hit_trough")]
    for r, q in zip(prim, bh([r["p"] for r in prim])): r["q_bh"] = q
    u25 = {c: row(session_D(d, "U25", c)[0], family="replication", basis="U25", stat=c) for c in ("hit_peak", "hit_trough")}
    ok = {c: (prim[i]["hl"] > 0 or prim[i]["n_pos"] > prim[i]["n_neg"]) and prim[i]["q_bh"] < 0.05 and u25[c]["p"] < 0.05 and u25[c]["n_pos"] > u25[c]["n_neg"]
          for i, c in enumerate(("hit_peak", "hit_trough"))}
    verdict = "supported" if all(ok.values()) else ("partly supported (" + ", ".join(k for k, v in ok.items() if v) + ")" if any(ok.values()) else "not supported")
    sec = []
    for b in ("STD",):
        for c in ("hit_peak", "hit_trough"): sec.append(row(session_D(d, b, c)[0], family="secondary", basis=b, stat=c))
    for b in ("U0", "U25", "STD"):
        for c in ("err_peak", "err_trough"):
            D = -session_D(d, b, c)[0]; sec.append(row(D, family="secondary", basis=b, stat=f"-{c} (rest minus reach error, ms)"))
        x = d[(d.basis == b) & (d.window == "reach")].groupby("session").peak_first.mean()
        sec.append(dict(family="secondary", basis=b, stat="peak precedes trough (session fraction > 0.5)", n=len(x), n_pos=int((x > 0.5).sum()), hl=float(x.median())))
    for name, sub in (("cohort2021", lambda x: x.session < 220000), ("cohort2022", lambda x: x.session >= 220000), ("stim0", lambda x: x.stim == 0)):
        for c in ("hit_peak", "hit_trough"): sec.append(row(session_D(d, "U0", c, sub)[0], family="sensitivity", basis="U0", stat=c, subset=name))
    tests = pd.DataFrame(prim + list(u25.values()) + sec); tests.to_csv(os.path.join(TAB, "e16_tests.csv"), index=False)
    hits = d.groupby(["basis", "window"])[["hit_peak", "hit_trough"]].mean().round(3); hits.to_csv(os.path.join(TAB, "e16_hit_rates.csv"))
    summ = dict(verdict=verdict, criteria_met=ok, n_conditions=int(d.key.nunique()), not_converged=int((~d.converged.astype(bool)).sum() // 3),
                primary={r["stat"]: {k: r[k] for k in ("hl", "ci95_lo", "ci95_hi", "n_pos", "n_neg", "p", "q_bh")} for r in prim},
                u25={c: {k: u25[c][k] for k in ("hl", "n_pos", "n_neg", "p")} for c in u25}, hit_rates=hits.reset_index().to_dict("records"))
    json.dump(summ, open(os.path.join(TAB, "e16_summary.json"), "w"), indent=1, default=float); print(json.dumps(summ, indent=1, default=float))
    figure(d)

def figure(d):
    tc = {}
    for f in sorted(glob.glob(os.path.join(SRC, "e16_ft_*.npz"))):
        z = np.load(f); tc.update({k: z[k] for k in z.files})
    off = np.arange(-400, 801); fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    meta = d[(d.window == "reach") & (d.basis == "U0")].set_index("key")
    for b, col in (("U0", "#1f4e99"), ("U25", "#2a9d8f"), ("STD", "#888")):
        M = []
        for k in meta.index:
            ft = tc[f"{k}__{b}"].astype(float); pv = int(meta.loc[k, "pv_ms"]); base = ft[pv - 1000:pv - 400]
            M.append((ft[pv + off] - base.mean()) / base.std())
        ses = meta.index.str[:6]; S = np.array([np.mean([m for m, s in zip(M, ses) if s == x], 0) for x in sorted(set(ses))])
        ax[0].plot(off, np.median(S, 0), color=col, lw=1.6 if b != "STD" else 1, label={"U0": "uniform 50-ms knots (primary)", "U25": "uniform, offset 25 ms", "STD": "standard basis (exploratory figure)"}[b])
    ax[0].axvline(0, color="k", ls=":", lw=.8); ax[0].axvline(float(np.median(meta.pa_ms - meta.pv_ms)), color="#c0392b", ls="--", lw=.8)
    ax[0].set_xlabel("time from peak velocity (ms)"); ax[0].set_ylabel("field term, baseline z (median over sessions)"); ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].set_title("a  1-ms field term (red dashed: median peak acceleration)", loc="left", fontsize=9)
    for j, c in enumerate(("hit_peak", "hit_trough")):
        a = ax[j + 1]
        for i, b in enumerate(("U0", "U25", "STD")):
            D, reach, rest = session_D(d, b, c)
            for k, (v, colr) in enumerate(((reach, "#1f4e99"), (rest, "#bbb"))):
                xx = i * 3 + k + np.random.default_rng(i * 2 + k).uniform(-.15, .15, len(v)); a.scatter(xx, v.values, s=12, color=colr, alpha=.8)
                a.plot([i * 3 + k - .3, i * 3 + k + .3], [v.mean()] * 2, color="k", lw=2)
        a.set_xticks([0, 1, 3, 4, 6, 7], ["U0\nreach", "U0\nrest", "U25\nreach", "U25\nrest", "STD\nreach", "STD\nrest"], fontsize=8)
        a.set_ylabel("fraction of conditions (session)"); a.set_title(("b  peak within ±30 ms of peak acceleration" if c == "hit_peak" else "c  trough within ±30 ms of peak velocity"), loc="left", fontsize=9)
    fig.text(.01, -.03, "One dot per session (mean of 3 conditions). Rest = same search with landmarks shifted by -2000 and -2500 ms. 5-ms smoothing.", fontsize=8, color="#555")
    fig.tight_layout()
    for e in ("png", "pdf"): fig.savefig(os.path.join(FIG, f"e16_onems_timing.{e}"), dpi=200, bbox_inches="tight")

def synth():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SRC, "synth_*.csv")))], ignore_index=True)
    assert d.key.nunique() == 6, d.key.nunique(); d.to_csv(os.path.join(TAB, "e16_synthetic_rows.csv"), index=False); out = []
    for (sc, b), g in d.groupby(["scenario", "basis"]):
        reach = g[g.window == "reach"].set_index("key"); rest = g[g.window.str.startswith("rest")].groupby("key")[["hit_peak", "hit_trough"]].mean()
        r = dict(scenario=sc, basis=b, peak_hits=int(reach.hit_peak.sum()), trough_hits=int(reach.hit_trough.sum()),
                 D_peak=float((reach.hit_peak - rest.hit_peak).mean()), D_trough=float((reach.hit_trough - rest.hit_trough).mean()),
                 med_err_peak=float(reach.err_peak.median()), med_err_trough=float(reach.err_trough.median()))
        if b == "U0": r["pass"] = (abs(r["D_peak"]) <= 0.2 and abs(r["D_trough"]) <= 0.2) if sc == "Z0" else (r["peak_hits"] >= 4 and r["trough_hits"] >= 4)
        out.append(r)
    T = pd.DataFrame(out); T.to_csv(os.path.join(TAB, "e16_synthetic_validation.csv"), index=False); print(T.to_string(index=False))

{"real": real, "synth": synth}[MODE]()
