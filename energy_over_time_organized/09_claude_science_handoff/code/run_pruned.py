"""Robustness: refit M0/M1 CV after dropping one unit of every pair with r > 0.5 (greedy, drop the lower-rate unit)."""
import sys, json, numpy as np, pandas as pd, multiprocessing as mp
sys.path.insert(0, ".")
import tvfield_model as tv
Z = dict(np.load(sys.argv[1])); LM = pd.read_csv(sys.argv[2], dtype={"session": str}); JOBS = json.load(open(sys.argv[3]))


def run(job):
    s, st, drop = job["session"], int(job["stim"]), job["drop"]; k = f"{s}_{st}"
    X = Z[f"X_{k}"]; fold = Z[f"fold_{k}"]; keep = [i for i in range(X.shape[2]) if i not in drop]; X = X[:, :, keep]
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0]); B = tv.bspline_basis(zc)
    g = np.zeros(X.shape[:2])
    for f in range(5):
        tr, te = fold != f, fold == f
        g[te] = tv.pseudo_ll(X[te], tv.fit(X[tr], B, "M1")) - tv.pseudo_ll(X[te], tv.fit(X[tr], B, "M0"))
    m = g.mean(0)
    return dict(session=s, stim=st, N_kept=len(keep), gain_all=m.mean(), gain_baseline=m[zc - 100:zc - 40].mean(),
                gain_late=m[zc + 50:zc + 150].mean(), gain_reach=m[zc - 50:zc + 50].mean())


with mp.get_context("fork").Pool(12) as p:
    res = p.map(run, JOBS)
pd.DataFrame(res).to_csv("pruned_gain.csv", index=False); print("done")
