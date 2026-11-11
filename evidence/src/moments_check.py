"""EXPLORATORY preview (in-sample): observed vs predicted statistics for the time-varying independent (M0) and pairwise (M1) models.
    python moments_check.py CODE_DIR spikes_v2.npz phase_landmarks.csv OUT_DIR SESSION
Per condition (exclusions applied), both models fitted to all reaches:
  - synchrony ratio per 10-ms bin: Var(K)/sum_i var_i, K = number of spiking cells (data: across reaches; models: from the model at that bin);
  - pooled (over all bins and reaches) connected pairwise and triplet correlations of spike indicators (0/1);
  - NOISE correlations: moments of the fluctuations around each cell's time-varying mean (data: PSTH over reaches, with the
    small-sample corrections R/(R-1) and R^2/((R-1)(R-2)); model: per-bin model mean), pooled over bins. The independent
    model predicts zero for these, so they isolate what the couplings must explain.
M1 moments from parallel Gibbs sampling per bin (spins +-1, P ~ exp(h.s + 1/2 s.J.s))."""
import sys, os
import numpy as np, pandas as pd
from scipy.special import expit
sys.path.insert(0, sys.argv[1]); import tvfield_model as tv
Z = dict(np.load(sys.argv[2])); LM = pd.read_csv(sys.argv[3]); OUT = sys.argv[4]; SESS = sys.argv[5]; os.makedirs(OUT, exist_ok=True)
CHAINS, BURN, KEEP, GAP = 200, 150, 10, 10

def pooled_from_samples(gen, N):
    """Accumulate pooled moments E[x_i], E[x_i x_j], E[x_i x_j x_k] from batches x (n, N) of 0/1 samples."""
    n = 0; m1 = np.zeros(N); m2 = np.zeros((N, N)); m3 = np.zeros((N, N, N))
    for x in gen:
        n += len(x); m1 += x.sum(0); m2 += x.T @ x; m3 += np.einsum("ni,nj,nk->ijk", x, x, x, optimize=True)
    return m1 / n, m2 / n, m3 / n

def connected(m1, m2, m3):
    N = len(m1); iu = np.triu_indices(N, 1); C2 = (m2 - np.outer(m1, m1))[iu]
    i, j, k = np.array([(a, b, c) for a in range(N) for b in range(a + 1, N) for c in range(b + 1, N)]).T if N >= 3 else (np.array([], int),) * 3
    C3 = m3[i, j, k] - m2[i, j] * m1[k] - m2[i, k] * m1[j] - m2[j, k] * m1[i] + 2 * m1[i] * m1[j] * m1[k]
    return C2, C3

def one(key):
    s, st = map(int, key.split("_")); X = Z["X_" + key][Z["exclude_" + key] == 0].astype(np.uint8); R, T, N = X.shape
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0]); B = tv.bspline_basis(zc)
    m0 = tv.fit(X, B, "M0"); m1 = tv.fit(X, B, "M1"); x = X.astype(float)
    # data
    K = x.sum(2); sync_d = K.var(0) / np.maximum(x.var(0).sum(1), 1e-12)
    D = connected(*pooled_from_samples((x[r] for r in range(R)), N))
    # M0 (independent within bin)
    p = expit(2 * m0["h"]); v0 = (p * (1 - p)).sum(1); sync_0 = np.ones(T)
    e1 = p.mean(0); e2 = np.einsum("ti,tj->ij", p, p) / T; np.fill_diagonal(e2, e1)
    e3 = np.einsum("ti,tj,tk->ijk", p, p, p, optimize=True) / T
    M0 = connected(e1, e2, e3)
    # M1 by Gibbs, independent chains per bin
    rng = np.random.default_rng([s, st]); h, J = m1["h"], m1["J"]
    S = np.where(rng.random((T, CHAINS, N)) < expit(2 * h)[:, None, :], 1.0, -1.0); snaps = []
    for sw in range(BURN + KEEP * GAP):
        for i in range(N): S[:, :, i] = np.where(rng.random((T, CHAINS)) < expit(2 * (h[:, None, i] + S @ J[:, i])), 1.0, -1.0)
        if sw >= BURN and (sw - BURN) % GAP == 0: snaps.append((S > 0).astype(np.float32).copy())
    Y = np.concatenate(snaps, 1)                                   # (T, CHAINS*KEEP, N)
    Ky = Y.sum(2); sync_1 = Ky.var(1) / np.maximum(Y.var(1).sum(1), 1e-12)
    M1 = connected(*pooled_from_samples((Y[t] for t in range(T)), N))
    # noise (fluctuation) correlations, pooled over bins
    iu = np.triu_indices(N, 1); tri = np.array([(a, b, c) for a in range(N) for b in range(a + 1, N) for c in range(b + 1, N)]).reshape(-1, 3)
    def noise_mom(dev_iter, n):
        c2 = np.zeros((N, N)); c3 = np.zeros(len(tri))
        for dv in dev_iter:
            c2 += dv.T @ dv
            if len(tri): c3 += (dv[:, tri[:, 0]] * dv[:, tri[:, 1]] * dv[:, tri[:, 2]]).sum(0)
        return c2[iu] / n, c3 / n
    dd = x - x.mean(0, keepdims=True)
    N2d, N3d = noise_mom((dd[:, t] for t in range(T)), R * T)
    N2d = N2d * R / (R - 1); N3d = N3d * R ** 2 / ((R - 1) * (R - 2))
    dy = Y - Y.mean(1, keepdims=True)
    N2m, N3m = noise_mom((dy[t] for t in range(T)), Y.shape[1] * T)
    sh = lambda a: np.roll(a, 400 - zc)                            # peak velocity at index 400
    return key, dict(sync_d=sh(sync_d), sync_0=sh(sync_0), sync_1=sh(sync_1), C2_d=D[0], C2_0=M0[0], C2_1=M1[0], C3_d=D[1], C3_0=M0[1], C3_1=M1[1], N2_d=N2d, N2_1=N2m, N3_d=N3d, N3_1=N3m)

if __name__ == "__main__":
    out = {}
    for k in sorted(x[2:] for x in Z if x.startswith("X_") and x[2:].split("_")[0] == SESS):
        key, d = one(k); out.update({f"{key}__{n}": v for n, v in d.items()}); print("done", key, flush=True)
    np.savez_compressed(os.path.join(OUT, f"moments_{SESS}.npz"), **out)
