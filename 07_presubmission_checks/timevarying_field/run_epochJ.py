"""Does coupling change during the reach? Per neuron, conditional logistic model with time-varying fields
(B-spline basis) and coupling weights that may differ by epoch:
  logit P(s_i=+1 | s_-i, t) = B(t) w + sum_j (w_j + d_reach_j 1[reach] + d_late_j 1[late]) s_j
reach = peak velocity -500..+500 ms, late = +500..+1500 ms. Compared on held-out reaches (same 5 folds)
against the shared-coupling version (d = 0). Conditional (pseudo-)likelihood, asymmetric weights."""
import sys, numpy as np, pandas as pd, multiprocessing as mp
sys.path.insert(0, ".")
import tvfield_model as tv
from scipy.special import expit

Z = dict(np.load(sys.argv[1])); LM = pd.read_csv(sys.argv[2], dtype={"session": str}); NCPU = int(sys.argv[3])
keys = sorted({k.split("_", 1)[1] for k in Z if k.startswith("X_")})
LAM, LAMB, T = 1e-3, 1e-5, 700


def cond_ll(F, y, w):
    z = F @ w
    return -(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))


def run(k):
    s, st = k.split("_"); st = int(st); X = Z[f"X_{k}"]; fold = Z[f"fold_{k}"]; R, _, N = X.shape
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0]); B = tv.bspline_basis(zc); K = B.shape[1]
    ind_r = np.zeros(T); ind_r[zc - 50:zc + 50] = 1; ind_l = np.zeros(T); ind_l[zc + 50:zc + 150] = 1
    S = 2.0 * X - 1
    ll = {m: np.zeros((R, T)) for m in ("shared", "epoch")}
    dJ = []
    for f in range(5):
        tr, te = fold != f, fold == f
        for i in range(N):
            oth = np.delete(np.arange(N), i)
            def feats(idx):
                r = idx.sum(); So = S[idx][:, :, oth].reshape(-1, N - 1)
                Bt = np.tile(B, (r, 1)); ir = np.tile(ind_r, r)[:, None]; il = np.tile(ind_l, r)[:, None]
                return np.hstack([Bt, So]), np.hstack([Bt, So, So * ir, So * il]), X[idx][:, :, i].reshape(-1).astype(float)
            F1, F2, y = feats(tr); G1, G2, yt = feats(te)
            w1 = tv._fit_logistic(F1, y, np.concatenate([np.full(K, LAMB), np.full(N - 1, LAM)]))
            w2 = tv._fit_logistic(F2, y, np.concatenate([np.full(K, LAMB), np.full(3 * (N - 1), LAM)]))
            ll["shared"][te] += cond_ll(G1, yt, w1).reshape(-1, T)
            ll["epoch"][te] += cond_ll(G2, yt, w2).reshape(-1, T)
            if f == 0:
                dJ.append(dict(i=i, base=w2[K:K + N - 1].mean() / 2, d_reach=w2[K + N - 1:K + 2 * (N - 1)].mean() / 2,
                               d_late=w2[K + 2 * (N - 1):].mean() / 2, abs_base=np.abs(w2[K:K + N - 1]).mean() / 2))
    d = (ll["epoch"] - ll["shared"]).mean(0)
    dj = pd.DataFrame(dJ).mean()
    return dict(session=s, stim=st, N=N, zc=zc, gain_all=d.mean(), gain_reach=d[zc - 50:zc + 50].mean(), gain_late=d[zc + 50:zc + 150].mean(),
                gain_base=d[zc - 150:zc - 50].mean(), meanJ_base=dj.base, mean_dJ_reach=dj.d_reach, mean_dJ_late=dj.d_late, mean_absJ_base=dj.abs_base)


with mp.get_context("fork").Pool(NCPU) as pool:
    res = []
    for n, r in enumerate(pool.imap_unordered(run, keys)):
        res.append(r); print(n + 1, r["session"], r["stim"], flush=True)
pd.DataFrame(res).to_csv("epochJ_results.csv", index=False)
print("done")
