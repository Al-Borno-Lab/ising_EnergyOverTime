"""E14 library (design: PREREGISTRATION_2026-10-27.md).
Spins s in {-1,+1}; P(s | r, t) ~ exp(sum_i h_i(t) s_i + 1/2 s.J_eff(r,t).s), J_eff(r,t) = J + 1[(r,t) in window] * sum_g kappa_g J^(g),
where J^(g) = J restricted to pair class g (0 = other pairs, 1 = suppressed pairs). PL fits as tvfield_model (lamB 1e-5, lamJ 1e-3)."""
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.special import expit

TOTAL_MS, CROSS_MS = 7000, 4000
FINE_MS, FINE_STEP_MS, COARSE_STEP_MS = (-600, 1600), 50, 250
NBASIS = 67
LAMB, LAMJ = 1e-5, 1e-3
BIN_SIZES = (1, 5, 10, 20)
WIN_MS = {"B": ("cross", -1000, -400), "E": ("end", -100, 0), "A": ("cross", -90, 0)}   # [start, stop) ms relative to anchor
NSHUF, SEED = 20, 14
GIBBS_BURN = 200

# ---------------------------------------------------------------- data
def load_block(npz, stim):
    """Binarized 1-ms data of one block, exclusions applied. Returns X (R,7000,N) uint8, cross_ms (R,), end_ms (R,)."""
    m = (npz["stim"] == stim) & (npz["exclude"] == 0)
    X = (npz["c1"][m] > 0).astype(np.uint8)
    assert X.shape[1] == TOTAL_MS
    return X, npz["idx_cross"][m] * 10.0, npz["idx_end"][m] * 10.0

def rebin(X1, b):
    """1-ms binary -> b-ms binary (any spike in the bin)."""
    R, T, N = X1.shape; assert T % b == 0
    return (X1.reshape(R, T // b, b, N).max(2)).astype(np.uint8)

def basis(b):
    """B-spline basis with knots fixed in ms (tvfield_model construction), centred on the 1-cm crossing. (T_b, 67)."""
    T = TOTAL_MS // b; c = CROSS_MS / b; a, e = c + FINE_MS[0] / b, c + FINE_MS[1] / b
    kn = np.unique(np.concatenate([np.arange(0, a, COARSE_STEP_MS / b), np.arange(a, e, FINE_STEP_MS / b),
                                   np.arange(e, T, COARSE_STEP_MS / b), [T - 1]]))
    t = np.concatenate([[kn[0]] * 3, kn, [kn[-1]] * 3]); nb = len(t) - 4
    B = np.column_stack([BSpline(t, np.eye(nb)[i], 3, extrapolate=False)(np.arange(T, dtype=float)) for i in range(nb)])
    B = np.nan_to_num(B)
    if B[-1].sum() == 0: B[-1] = B[-2]
    assert B.shape == (T, NBASIS), B.shape
    return B

def window_bins(anchor_ms, name, b):
    """Bins (at size b) lying wholly inside window `name` for one reach; None if the anchor is not finite or out of range."""
    which, s, e = WIN_MS[name]
    if not np.isfinite(anchor_ms): return None
    lo, hi = anchor_ms + s, anchor_ms + e
    if lo < 0 or hi > TOTAL_MS: return None
    return np.arange(int(np.ceil(lo / b)), int(np.floor(hi / b)))

def window_mask(cross_ms, end_ms, name, b, R):
    T = TOTAL_MS // b; M = np.zeros((R, T), bool)
    for r in range(R):
        anc = cross_ms[r] if WIN_MS[name][0] == "cross" else end_ms[r]
        w = window_bins(anc, name, b)
        if w is not None: M[r, w] = True
    return M

# ---------------------------------------------------------------- M1 fit (pseudo-likelihood; copy of the S3 1-ms code, tvfield_model math)
G = {}

def _stage1(i):
    X, S, Bt, K = G["X"], G["S"], G["Bt"], G["K"]; R, T, N = X.shape
    y = X[:, :, i].reshape(-1).astype(float); oth = np.delete(np.arange(N), i)
    F = np.hstack([Bt, S[:, :, oth].reshape(-1, N - 1)]); pen = np.concatenate([np.full(K, LAMB), np.full(N - 1, LAMJ)]); n = len(y)
    def f(w):
        z = F @ w; return (np.logaddexp(0, z) - y * z).mean() + (pen * w * w).sum(), F.T @ (expit(z) - y) / n + 2 * pen * w
    r = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return r.x, bool(r.success)

def _stage2(args):
    i, Ji, h0 = args; X, S, Bt, B = G["X"], G["S"], G["Bt"], G["B"]
    y = X[:, :, i].reshape(-1).astype(float); off = (S @ Ji).reshape(-1)
    def f(wb):
        z = Bt @ wb + 2 * off
        return (np.logaddexp(0, z) - y * z).mean() + LAMB * (wb * wb).sum(), Bt.T @ (expit(z) - y) / len(y) + 2 * LAMB * wb
    r = minimize(f, 2 * np.linalg.lstsq(B, h0, rcond=None)[0], jac=True, method="L-BFGS-B")
    return B @ r.x / 2, bool(r.success)

def fit_M1(X, B, ncpu):
    R, T, N = X.shape; K = B.shape[1]
    G.update(X=X, S=2.0 * X - 1, Bt=np.tile(B, (R, 1)), B=B, K=K)
    with Pool(min(ncpu, N)) as p: out = p.map(_stage1, range(N))
    H = np.zeros((T, N)); W = np.zeros((N, N)); ok = []
    for i, (w, s) in enumerate(out):
        H[:, i] = B @ w[:K] / 2; W[i, np.delete(np.arange(N), i)] = w[K:] / 2; ok.append(s)
    J = (W + W.T) / 2
    with Pool(min(ncpu, N)) as p: out = p.map(_stage2, [(i, J[i], H[:, i]) for i in range(N)])
    for i, (h, s) in enumerate(out): H[:, i] = h; ok.append(s)
    G.clear()
    return dict(h=H, J=J, converged=bool(all(ok)))

# ---------------------------------------------------------------- epoch coupling scale (joint PL over fields + kappa, J fixed)
def fit_kappa(X, B, J, H0, mask, cls, maxiter=2000):
    """X (R,T,N); mask (R,T) bool = window; cls (N,N) int pair class (0 other, 1 suppressed; diagonal ignored).
    Joint penalised PL over all field weights and kappa_g (no penalty on kappa). Classes with no pairs keep kappa = 0 and return NaN."""
    R, T, N = X.shape; K = B.shape[1]; S = 2.0 * X - 1
    Bt = np.tile(B, (R, 1)); n = R * T; m = mask.reshape(-1).astype(float)
    off = np.eye(N, dtype=bool)
    Jg = [np.where((cls == g) & ~off, J, 0.0) for g in (0, 1)]
    present = [bool(np.any(Jg[g] != 0)) for g in (0, 1)]
    Y = X.reshape(n, N).astype(float); Sf = S.reshape(n, N)
    Ctot = Sf @ J                                             # (n, N): sum_j J_ij s_j (J symmetric, zero diagonal)
    Cg = [Sf @ Jg[g] for g in (0, 1)]
    Wb0 = np.linalg.lstsq(B, H0, rcond=None)[0] * 2          # (K, N) logistic-scale weights
    def unpack(p): return p[:K * N].reshape(K, N), p[K * N:]
    def f(p):
        Wb, kap = unpack(p); kap = np.where(present, kap, 0.0)
        Z = Bt @ Wb + 2 * (Ctot + m[:, None] * (kap[0] * Cg[0] + kap[1] * Cg[1]))
        Rres = expit(Z) - Y
        loss = (np.logaddexp(0, Z) - Y * Z).sum() / n + LAMB * (Wb * Wb).sum()
        gW = Bt.T @ Rres / n + 2 * LAMB * Wb
        gk = np.array([(Rres * (2 * m[:, None] * Cg[g])).sum() / n if present[g] else 0.0 for g in (0, 1)])
        return loss, np.concatenate([gW.ravel(), gk])
    r = minimize(f, np.concatenate([Wb0.ravel(), [0.0, 0.0]]), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    Wb, kap = unpack(r.x)
    return dict(kappa=[float(kap[g]) if present[g] else float("nan") for g in (0, 1)], h=B @ Wb / 2,
                converged=bool(r.success), nit=int(r.nit), n_pairs=[int(np.triu((cls == g) & ~off, 1).sum()) for g in (0, 1)])

# ---------------------------------------------------------------- synchrony index (1 ms)
WIN_LEN_1MS = {"B": 600, "E": 100, "A": 90}

def windows_fixed(X, cross_ms, end_ms, test="E"):
    """1-ms arrays for window B and the test window (E or A) over reaches valid for both. Returns XB, Xtest, kept reach indices."""
    keep, iB, iE = [], [], []
    for r in range(X.shape[0]):
        b = window_bins(cross_ms[r], "B", 1); e = window_bins(cross_ms[r] if WIN_MS[test][0] == "cross" else end_ms[r], test, 1)
        if b is None or e is None: continue
        assert len(b) == WIN_LEN_1MS["B"] and len(e) == WIN_LEN_1MS[test]
        keep.append(r); iB.append(b); iE.append(e)
    keep = np.array(keep, int)
    return X[keep[:, None], np.array(iB)], X[keep[:, None], np.array(iE)], keep

def suppressed(XB, XE):
    """Cell class: mean spike probability in E < in B."""
    return XE.mean((0, 1)) < XB.mean((0, 1))

def delta_si(Xw, rng):
    """Shuffle-corrected SI matrix for one window. Xw (R, L, N) binary. NaN where a cell is silent."""
    R, L, N = Xw.shape; x = Xw.astype(float); p = x.mean((0, 1))
    C = np.einsum("rti,rtj->ij", x, x) / (R * L)
    Cs = np.zeros((N, N))
    for _ in range(NSHUF):
        while True:
            pi = rng.permutation(R)
            if R < 2 or np.all(pi != np.arange(R)): break
        c = np.einsum("rti,rtj->ij", x, x[pi]) / (R * L); Cs += (c + c.T) / 2
    Cs /= NSHUF
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.outer(p, p); out = (C - Cs) / den
    out[den == 0] = np.nan
    return out

def pair_D(XB, XE, sup, rng):
    """Per-pair D = dSI_E - dSI_B, and class means. Returns dict(D_supp, D_other, D_all, n_supp, n_other)."""
    D = delta_si(XE, rng) - delta_si(XB, rng); N = D.shape[0]
    iu = np.triu_indices(N, 1); d = D[iu]; s = sup[iu[0]] & sup[iu[1]]; ok = np.isfinite(d)
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    return dict(D_supp=mean(d[ok & s]), D_other=mean(d[ok & ~s]), D_all=mean(d[ok]), n_supp=int((ok & s).sum()), n_other=int((ok & ~s).sum()))

def pair_classes(sup):
    return (sup[:, None] & sup[None, :]).astype(int)

# ---------------------------------------------------------------- Gibbs sampling (independent draw per unit = (reach, bin))
def gibbs_units(Hu, J, rng, sweeps=GIBBS_BURN, mu=None, Jextra=None):
    """Hu (U, N) fields per unit; J (N,N). Optional mu (U,) multiplier on Jextra (window coupling change). Returns binary (U, N)."""
    U, N = Hu.shape; S = np.where(rng.random((U, N)) < expit(2 * Hu), 1.0, -1.0)
    for _ in range(sweeps):
        for i in range(N):
            loc = Hu[:, i] + S @ J[:, i]
            if Jextra is not None: loc = loc + mu * (S @ Jextra[:, i])
            S[:, i] = np.where(rng.random(U) < expit(2 * loc), 1.0, -1.0)
    return (S > 0).astype(np.uint8)

def simulate(h, J, R, rng, mask=None, Jextra=None, gain_sd=0.0):
    """Synthetic (R, T, N) data from fields h (T,N): per (reach, bin) independent Gibbs draws. Inside `mask` (R,T) the coupling is
    J + Jextra and, if gain_sd > 0, every field gets a per-reach offset delta_r ~ N(0, gain_sd) (T2)."""
    T, N = h.shape; Hu = np.tile(h, (R, 1)).astype(float); mu = None
    if mask is not None:
        mu = mask.reshape(-1).astype(float)
        if gain_sd > 0: Hu += (mu * np.repeat(rng.normal(0, gain_sd, R), T))[:, None]
    return gibbs_units(Hu, J, rng, mu=mu, Jextra=Jextra).reshape(R, T, N)

def predicted_D(h1, J, XB, XE, keep, cross_ms, end_ms, sup, rng, reps=5):
    """N3: simulate the fixed-J 1-ms model at each kept reach's B and E bins, compute D for the real cell classes; mean over reps."""
    out = []
    for _ in range(reps):
        iB = np.array([window_bins(cross_ms[r], "B", 1) for r in keep]); iE = np.array([window_bins(end_ms[r], "E", 1) for r in keep])
        sB = gibbs_units(h1[iB.reshape(-1)], J, rng).reshape(XB.shape); sE = gibbs_units(h1[iE.reshape(-1)], J, rng).reshape(XE.shape)
        out.append(pair_D(sB, sE, sup, rng))
    return {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}
