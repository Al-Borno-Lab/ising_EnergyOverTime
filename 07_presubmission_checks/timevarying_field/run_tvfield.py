
import sys, numpy as np, pandas as pd, time
import multiprocessing as mp
sys.path.insert(0, ".")
import tvfield_model as tv
Z = dict(np.load(sys.argv[1])); Z["files"]=None; LM = pd.read_csv(sys.argv[2], dtype={"session": str}); NSH = int(sys.argv[3]); NCPU = int(sys.argv[4])
keys = sorted({k.split("_", 1)[1] for k in Z if k.startswith("X_")})
def run(k):
    s, st = k.split("_"); st = int(st); X = Z[f"X_{k}"]; fold = Z[f"fold_{k}"]; R, T, N = X.shape
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0]); B = tv.bspline_basis(zc)
    rng = np.random.default_rng(1000 + st * 100 + int(s) % 97)
    out = {}
    def cv(Xc, models, exact):
        res = {m: np.full((R, T), np.nan) for m in models}; resx = {m: np.full((R, T), np.nan) for m in models if exact and m in ("M0", "M1")}
        for f in range(5):
            tr, te = fold != f, fold == f
            for m in models:
                P = tv.fit(Xc[tr], B, m) if m != "Ms" else tv.fit(Xc[tr], np.ones((T, 1)), "Ms")
                res[m][te] = tv.pseudo_ll(Xc[te], P)
                if m in resx: resx[m][te] = tv.exact_ll(Xc[te], P)
        return res, resx
    t0 = time.time()
    pl, ex = cv(X, ["M0", "M1", "Mlag", "Ms"], exact=(N <= 20))
    out["pl"] = pl; out["ex"] = ex
    null = []
    for b in range(NSH):
        Xs = tv.trial_shuffle(X, rng); r_, _ = cv(Xs, ["M0", "M1"], exact=False)
        null.append((r_["M1"] - r_["M0"]).mean(0))
    out["null"] = np.array(null)
    full = tv.fit(X, B, "M1"); out["full_h"] = full["h"]; out["full_J"] = full["J"]
    out["meta"] = dict(session=s, stim=st, N=N, R=R, zc=zc, secs=time.time() - t0)
    return k, out
ctx = mp.get_context('fork')
with ctx.Pool(NCPU) as pool:
    res = []
    for i, r in enumerate(pool.imap_unordered(run, keys)):
        res.append(r); print(i + 1, r[0], round(r[1]['meta']['secs']), flush=True)
import pickle; pickle.dump(dict(res), open("tvfield_cv_raw.pkl", "wb"))
print("done", len(res))
