"""E13 statistics from <dir>/ll_<cond>.npz and the landmarks table. Session-level (n = 21) tests G1, G2, G3a, G3b; BH over {G2, G3a x3, G3b}."""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import glm_lib as G

MOUSE = {**{s: "fChR2" for s in (210421, 210422, 210423, 210425, 210514)}, **{s: "fChR5" for s in (210511, 210512, 210515)},
         **{s: "fChR4" for s in (210606, 210608, 210614)}, **{s: "cChR1" for s in (210619, 210620, 210622, 210623)},
         **{s: "DJC002" for s in (220515, 220517, 220519)}, **{s: "DJC000" for s in (220516, 220518, 220520)}}
rng = np.random.RandomState(13)
GAIN_MODELS = ["M1", "SYNC", "LAG1", "LAG1H", "SYNC_LAG1", "SYNC_H", "LAG2", "MLAG20", "SYNC_E"]


def cond_table(d, lm):
    rows = []
    for f in sorted(glob.glob(f"{d}/ll_*.npz")):
        key = os.path.basename(f)[3:-4]; s, st = key.split("_")
        r = lm[(lm.session == int(s)) & (lm.stim == int(st))].iloc[0]
        sl = G.phase_slices(r); sl["all"] = slice(0, G.T)
        z = np.load(f)
        ll0 = z["ll_M0"]
        for p, sli in sl.items():
            row = dict(cond=key, session=int(s), stim=int(st), phase=p, ll_M0=np.nanmean(ll0[:, sli]))
            for m in GAIN_MODELS:
                if f"ll_{m}" in z: row[f"gain_{m}"] = np.nanmean(z[f"ll_{m}"][:, sli] - ll0[:, sli])
            for m in ("SYNC", "LAG1H"):
                if f"null_{m}" in z: row[f"null_{m}"] = np.nanmean(z[f"null_{m}"][:, sli])
            if "ll_SYNC" in z and "ll_SYNC_E" in z: row["epoch_vs_shared"] = np.nanmean(z["ll_SYNC_E"][:, sli] - z["ll_SYNC"][:, sli])
            if "ll_SYNC" in z and "ll_M1" in z: row["sync_minus_M1"] = np.nanmean(z["ll_SYNC"][:, sli] - z["ll_M1"][:, sli])
            for m in ("SYNC", "LAG1H"):
                if f"null_{m}" in z: row[f"net_{m}"] = row[f"gain_{m}"] - row[f"null_{m}"]
            rows.append(row)
    ct = pd.DataFrame(rows)
    # J-constant reference (amendment A1): per phase mean of the simulated constant-J SYNC gain, if ref files exist
    refs = {}
    for f in glob.glob(f"{d}/ref_*.npz"):
        key = os.path.basename(f)[4:-4]; refs[key] = np.load(f)["refgain"]
    if refs:
        vals = []
        for _, r in ct.iterrows():
            if r.cond not in refs: vals.append(np.nan); continue
            s_, st_ = r.cond.split("_"); lr = lm[(lm.session == int(s_)) & (lm.stim == int(st_))].iloc[0]
            sl = G.phase_slices(lr); sl["all"] = slice(0, G.T)
            vals.append(float(np.nanmean(refs[r.cond][sl[r.phase]])))
        ct["ref_gain_SYNC"] = vals
    return ct


def hl(x):
    x = np.asarray(x); w = (x[:, None] + x[None]) / 2
    return float(np.median(w[np.triu_indices(len(x))]))


def boot(x, fn=np.mean, nb=10000):
    x = np.asarray(x, float); n = len(x)
    return np.percentile([fn(x[rng.randint(0, n, n)]) for _ in range(nb)], [2.5, 97.5])


def wil(x, null=0.0):
    x = np.asarray(x, float) - null; x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0): return 1.0
    return float(stats.wilcoxon(x).pvalue)


def mouse_fit(series):
    df = pd.DataFrame(dict(d=series.values, mouse=[MOUSE[i] for i in series.index]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("d ~ 1", df, groups=df.mouse).fit(reml=True)
        return float(m.params["Intercept"]), *[float(v) for v in m.conf_int().loc["Intercept"]]
    except Exception:
        return (np.nan,) * 3


def summarize(name, x, null=0.0, fn=np.mean):
    x = x.dropna(); lo, hi = boot(x.values, fn)
    out = dict(test=name, n=len(x), est=float(fn(x.values)), ci_lo=lo, ci_hi=hi, median=float(np.median(x)), p=wil(x.values, null),
               n_above=int((x.values > null).sum()))
    for coh in (2021, 2022):
        xc = x[[(i < 220000) == (coh == 2021) for i in x.index]]
        out[f"p_{coh}"] = wil(xc.values, null) if len(xc) > 1 else np.nan; out[f"est_{coh}"] = float(fn(xc.values)) if len(xc) else np.nan
    out["mm_est"], out["mm_lo"], out["mm_hi"] = mouse_fit(x)
    return out


def run(d, lm_path, out):
    os.makedirs(out, exist_ok=True)
    lm = pd.read_csv(lm_path)
    ct = cond_table(d, lm); ct.to_csv(f"{out}/condition_phase_table.csv", index=False)
    s = ct.groupby(["session", "phase"]).mean(numeric_only=True).reset_index()
    s.to_csv(f"{out}/session_phase_table.csv", index=False)
    A = s[s.phase == "all"].set_index("session")
    tests = []
    # G1 (not in family)
    g1 = summarize("G1: LL_SYNC - LL_M1 (nats/bin)", A.sync_minus_M1); g1["equivalent_pm0.01"] = bool(-0.01 <= g1["ci_lo"] and g1["ci_hi"] <= 0.01)
    # gains table (all bins)
    gains = pd.DataFrame({m: A[f"gain_{m}"] for m in GAIN_MODELS if f"gain_{m}" in A})
    for m in ("SYNC", "LAG1H"):
        if f"net_{m}" in A: gains[f"net_{m}"] = A[f"net_{m}"]
    gains.describe().T.to_csv(f"{out}/gain_summary_all_bins.csv")
    # G2
    r = (A.net_LAG1H / A.net_SYNC)
    g2 = summarize("G2: net gain LAG1H / net gain SYNC (< 0.5?)", r, null=0.5, fn=np.median); g2["pooled_ratio_of_means"] = float(A.net_LAG1H.mean() / A.net_SYNC.mean())
    g2["decision_below_half"] = bool(g2["ci_hi"] < 0.5)
    sec = []
    for m in ("LAG1", "LAG2", "MLAG20", "SYNC_LAG1", "SYNC_H", "M1"):
        if f"gain_{m}" in A: sec.append(dict(model=m, median_gain=A[f"gain_{m}"].median(), median_ratio_to_SYNC_raw=(A[f"gain_{m}"] / A.gain_SYNC).median(), n_sessions_gain_pos=int((A[f"gain_{m}"] > 0).sum())))
    pd.DataFrame(sec).to_csv(f"{out}/secondary_raw_gain_ratios.csv", index=False)
    # G3a (net gain change, descriptive when a J-constant reference exists) and G3a' (observed minus J-constant reference; primary, A1)
    P = {p: s[s.phase == p].set_index("session") for p in G.PHASES}
    g3a = []
    for p in ("acceleration", "deceleration", "post"):
        ch = P[p].net_SYNC - P["baseline"].net_SYNC
        t_ = summarize(f"G3a: SYNC net gain {p} - baseline (nats/bin)", ch)
        rel = []; n = len(ch)
        for _ in range(10000):
            ix = rng.randint(0, n, n); rel.append(P[p].net_SYNC.values[ix].mean() / P["baseline"].net_SYNC.values[ix].mean() - 1)
        t_["rel_change"] = P[p].net_SYNC.mean() / P["baseline"].net_SYNC.mean() - 1; t_["rel_lo"], t_["rel_hi"] = np.percentile(rel, [2.5, 97.5])
        g3a.append(t_)
    pd.DataFrame(g3a).to_csv(f"{out}/G3a_net_gain_change_descriptive.csv", index=False)
    fam = [g2]
    if "ref_gain_SYNC" in s and s.ref_gain_SYNC.notna().any():
        for p in ("acceleration", "deceleration", "post"):
            obs = P[p].gain_SYNC - P["baseline"].gain_SYNC; ref = P[p].ref_gain_SYNC - P["baseline"].ref_gain_SYNC
            t_ = summarize(f"G3a': SYNC raw-gain change {p} - baseline, observed minus J-constant reference (nats/bin)", obs - ref)
            t_["obs_change"] = float(obs.mean()); t_["ref_change"] = float(ref.mean()); fam.append(t_)
    else:
        fam += g3a
    fam.append(summarize("G3b: held-out LL epoch-specific - shared weights (nats/bin)", A.epoch_vs_shared))
    fam = pd.DataFrame(fam); fam["p_bh"] = multipletests(fam.p, method="fdr_bh")[1]
    fam.to_csv(f"{out}/primary_family.csv", index=False); pd.DataFrame([g1]).to_csv(f"{out}/G1.csv", index=False)
    # net gains per phase for context
    pd.DataFrame({p: P[p][["net_SYNC", "net_LAG1H", "gain_SYNC", "gain_LAG1", "gain_M1"]].median() for p in G.PHASES}).T.to_csv(f"{out}/phase_median_gains.csv")
    return ct, s, fam, g1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ll", required=True); ap.add_argument("--landmarks", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); run(a.ll, a.landmarks, a.out)
