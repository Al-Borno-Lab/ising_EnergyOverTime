"""E13 G4 (descriptive): P(K) of data vs samples from M0, M1, SYNC (pseudo-Gibbs) and LAG1H (forward simulation), fitted on all kept reaches.
K = number of spikes in a 10-ms bin. Windows: baseline [zc-100, zc-40) and reach [zc-50, zc+50). Output: <out>/pk_<cond>.npz and pk_summary.csv (total-variation distance)."""
import argparse, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd
import tvfield_model as tv
import glm_lib as G

CH, SWEEPS = 3000, 40
WINDOWS = lambda zc: {"baseline": range(zc - 100, zc - 40), "reach": range(zc - 50, zc + 50)}


def fit_weights(X, B, zc, model):
    """Per-neuron weights: dict i -> (w_baseline (K,), blocks {name: w}). Returns logit-form parameters."""
    R, _, N = X.shape; S = 2.0 * X - 1; K = B.shape[1]; Bt = np.tile(B, (R, 1)); out = []
    for i in range(N):
        blk = G.feature_blocks(S, i, zc); names = [] if model == "M0" else G.SPEC[model]
        F = np.hstack([Bt] + [blk[n].reshape(R * G.T, -1) for n in names])
        pen = np.concatenate([np.full(K, G.LAMB), np.full(F.shape[1] - K, G.LAMJ)])
        w = tv._fit_logistic(F, X[:, :, i].reshape(-1).astype(float), pen)
        d = {"base": w[:K]}; o = K
        for n in names:
            dim = blk[n].shape[2]; d[n] = w[o:o + dim]; o += dim
        out.append(d)
    return out


def pk_hist(Xs, N):
    return np.bincount(Xs.sum(1).astype(int), minlength=N + 1) / len(Xs)


def gibbs_sync(W, B, N, bins, rng):
    """Pseudo-Gibbs for same-bin conditionals. W: per-neuron weights (SYNC) or M1 (given as logit form too)."""
    res = {}
    base = np.stack([B @ W[i]["base"] for i in range(N)], 1)            # (T,N) logits
    Wm = np.zeros((N, N))
    for i in range(N):
        Wm[i, np.delete(np.arange(N), i)] = W[i]["sync"]
    for t in bins:
        s = 2.0 * (rng.rand(CH, N) < 1 / (1 + np.exp(-base[t]))) - 1
        for _ in range(SWEEPS):
            for i in range(N):
                z = base[t, i] + s @ Wm[i]
                s[:, i] = np.where(rng.rand(CH) < 1 / (1 + np.exp(-z)), 1.0, -1.0)
        res[t] = ((s + 1) / 2)
    return res


def sim_lag1h(W, B, N, zc, rng, nch=300):
    base = np.stack([B @ W[i]["base"] for i in range(N)], 1)
    L = np.zeros((N, N))
    for i in range(N): L[i, np.delete(np.arange(N), i)] = W[i]["lag1"]
    H = np.stack([W[i]["hist"] for i in range(N)], 0)                    # (N,3)
    s_hist = np.zeros((nch, N, 3)); out = np.zeros((nch, G.T, N))        # hist in spins; zero-padded
    prev = np.zeros((nch, N))
    for t in range(G.T):
        z = base[t][None] + prev @ L.T + (s_hist * H[None]).sum(2)
        x = (rng.rand(nch, N) < 1 / (1 + np.exp(-z)))
        out[:, t] = x; sp = 2.0 * x - 1
        s_hist = np.concatenate([sp[:, :, None], s_hist[:, :, :2]], 2); prev = sp
    return out


def work(a):
    key, X, zc = a
    rng = np.random.RandomState((int(key.split("_")[0]) * 10 + int(key.split("_")[1])) % 2**31)
    B = tv.bspline_basis(zc); N = X.shape[2]; wins = WINDOWS(zc); allbins = sorted({t for w in wins.values() for t in w})
    d = {}
    for wn, w in wins.items():
        d[f"data_{wn}"] = pk_hist(np.concatenate([X[:, t, :] for t in w]), N)
    # M0
    W0 = fit_weights(X, B, zc, "M0"); p0 = 1 / (1 + np.exp(-np.stack([B @ W0[i]["base"] for i in range(N)], 1)))
    for wn, w in wins.items():
        d[f"M0_{wn}"] = pk_hist(np.concatenate([(rng.rand(3000, N) < p0[t]).astype(float) for t in w]), N)
    # M1 as Ising: Gibbs on exact conditionals
    P1 = tv.fit(X, B, "M1"); h, J = P1["h"], P1["J"]
    smp = {}
    for t in allbins:
        s = 2.0 * (rng.rand(CH, N) < 1 / (1 + np.exp(-2 * h[t]))) - 1
        for _ in range(SWEEPS):
            for i in range(N):
                z = 2 * (h[t, i] + s @ J[i]); s[:, i] = np.where(rng.rand(CH) < 1 / (1 + np.exp(-z)), 1.0, -1.0)
        smp[t] = (s + 1) / 2
    for wn, w in wins.items(): d[f"M1_{wn}"] = pk_hist(np.concatenate([smp[t] for t in w]), N)
    # SYNC pseudo-Gibbs
    Ws = fit_weights(X, B, zc, "SYNC"); sm = gibbs_sync(Ws, B, N, allbins, rng)
    for wn, w in wins.items(): d[f"SYNC_{wn}"] = pk_hist(np.concatenate([sm[t] for t in w]), N)
    # LAG1H forward simulation
    Wl = fit_weights(X, B, zc, "LAG1H"); sl = sim_lag1h(Wl, B, N, zc, rng)
    for wn, w in wins.items(): d[f"LAG1H_{wn}"] = pk_hist(np.concatenate([sl[:, t, :] for t in w]), N)
    return key, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--landmarks", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--only", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    z = np.load(a.npz, allow_pickle=True); lm = pd.read_csv(a.landmarks); jobs = []
    for _, r in lm.iterrows():
        key = f"{int(r.session)}_{int(r.stim)}"
        if a.only and key not in a.only.split(","): continue
        keep = z["exclude_" + key] == 0
        jobs.append((key, z["X_" + key][keep].astype(np.uint8), int(r.zero_cross)))
    rows = []
    with Pool(a.nproc) as p:
        for key, d in p.imap_unordered(work, jobs):
            np.savez_compressed(f"{a.out}/pk_{key}.npz", **d)
            for wn in ("baseline", "reach"):
                for m in ("M0", "M1", "SYNC", "LAG1H"):
                    rows.append(dict(cond=key, session=int(key.split("_")[0]), window=wn, model=m, tv=0.5 * np.abs(d[f"{m}_{wn}"] - d[f"data_{wn}"]).sum()))
            print(key, flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/pk_tv_distance.csv", index=False)
