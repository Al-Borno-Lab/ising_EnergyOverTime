"""Exact held-out log-likelihood of STATIC models (constant fields; Ising with constant J, and independent),
on the same 5 reach folds as tvfield_cv_raw_excl.pkl (excluded reaches dropped). N <= 20 only (exact).
Usage: python run_static_ll.py NCPU"""
import sys, os, pickle, numpy as np, multiprocessing as mp
sys.path.append(os.getcwd())
import thermo_model as tm

Z = dict(np.load("spikes_v2.npz"))
keys = sorted({k.split("_", 1)[1] for k in Z if k.startswith("X_")})
keys = [k for k in keys if Z[f"X_{k}"].shape[2] <= 20]


def run(k):
    keep = Z[f"exclude_{k}"] == 0; X = Z[f"X_{k}"][keep]; fold = Z[f"fold_{k}"][keep]; R, T, N = X.shape
    ll_is = np.full((R, T), np.nan); ll_in = np.full((R, T), np.nan)
    one = np.ones((1, 1))
    for f in range(5):
        tr, te = fold != f, fold == f
        Xtr = X[tr].reshape(-1, 1, N)                         # every bin = one sample, single time point
        F = tm.fit_exact(Xtr, one, lamB=1e-5, lamJ=1e-4)
        H = np.repeat(F["h"], T, 0)
        ll_is[te] = tm.loglik_bins(X[te], H, F["J"])
        p = (X[tr].sum((0, 1)) + 0.5) / (tr.sum() * T + 1)  # static independent, add-half smoothing
        s = X[te].astype(float)
        ll_in[te] = (s * np.log(p) + (1 - s) * np.log(1 - p)).sum(2)
    return k, dict(static_ising=ll_is, static_indep=ll_in, N=N, R=R)


if __name__ == "__main__":
    out = {}
    with mp.get_context("fork").Pool(int(sys.argv[1])) as pool:
        for i, (k, r) in enumerate(pool.imap_unordered(run, keys)):
            out[k] = r; print(i + 1, k, r["N"], flush=True)
    pickle.dump(out, open("static_ll_excl.pkl", "wb")); print("done", len(out))
