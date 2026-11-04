"""E16 library (design: PREREGISTRATION_2026-11-03.md). Self-contained: 1-ms M1 fit (pseudo-likelihood, tvfield_model math),
three B-spline bases, field term, peak/trough search, Gibbs simulation for the synthetic test.
Spins s in {-1,+1}; P(s|t) ~ exp(sum_i h_i(t) s_i + 1/2 s.J.s)."""
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

T = 7000
LAMB, LAMJ = 1e-5, 1e-3
STEP_MS, OFFSET_MS = 50, 25
STD_FINE, STD_COARSE = (-600, 1600), 250
SMOOTH_MS = 5
SEARCH = (-200, 100)            # ms relative to peak velocity
HIT_MS = 30
REST_SHIFTS = (-2000, -2500)
BASES = ("U0", "U25", "STD")
SEED = 16

def _bspline(kn):
    kn = np.unique(np.asarray(kn, float)); t = np.concatenate([[kn[0]] * 3, kn, [kn[-1]] * 3]); nb = len(t) - 4
    B = np.column_stack([BSpline(t, np.eye(nb)[i], 3, extrapolate=False)(np.arange(T, dtype=float)) for i in range(nb)])
    B = np.nan_to_num(B)
    if B[-1].sum() == 0: B[-1] = B[-2]
    return B

def basis(name, pv_ms):
    if name == "U0":  return _bspline(np.concatenate([np.arange(0, T, STEP_MS), [T - 1]]))
    if name == "U25": return _bspline(np.concatenate([[0], np.arange(OFFSET_MS, T, STEP_MS), [T - 1]]))
    if name == "STD":
        a, b = pv_ms + STD_FINE[0], pv_ms + STD_FINE[1]
        return _bspline(np.concatenate([np.arange(0, a, STD_COARSE), np.arange(a, b, STEP_MS), np.arange(b, T, STD_COARSE), [T - 1]]))
    raise ValueError(name)

# ---------------- M1 pseudo-likelihood fit (same math as tvfield_model.fit, parallel over neurons)
G = {}

def _stage1(i):
    X, S, Bt, K = G["X"], G["S"], G["Bt"], G["K"]; R, _, N = X.shape
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
    R, _, N = X.shape; K = B.shape[1]
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

# ---------------- field term and peak/trough search
def field_term(X, h):
    """FT(t) = -(1/N) mean_r sum_i h_i(t) s_i(r,t); s = +-1. Smoothed with a 5-ms Gaussian."""
    S = 2.0 * X - 1
    return gaussian_filter1d(-(S * h[None]).sum(2).mean(0) / X.shape[2], SMOOTH_MS)

def search(ft, pv, pa, shift=0):
    """Peak/trough times (ms relative to the shifted pv) in [pv+shift-200, pv+shift+100], and hits vs the shifted landmarks."""
    ref = pv + shift; w = np.arange(ref + SEARCH[0], ref + SEARCH[1] + 1); assert w[0] >= 0 and w[-1] < T
    tp = int(w[ft[w].argmax()] - ref); tt = int(w[ft[w].argmin()] - ref); pa_rel = pa - pv
    return dict(t_peak=tp, t_trough=tt, pa_rel=pa_rel, hit_peak=int(abs(tp - pa_rel) <= HIT_MS), hit_trough=int(abs(tt) <= HIT_MS),
                err_peak=abs(tp - pa_rel), err_trough=abs(tt), peak_first=int(tp < tt))

# ---------------- Gibbs simulation (independent draw per (reach, bin))
def simulate(h, J, R, rng, sweeps=200):
    N = h.shape[1]; Hu = np.tile(h, (R, 1)); S = np.where(rng.random(Hu.shape) < expit(2 * Hu), 1.0, -1.0)
    for _ in range(sweeps):
        for i in range(N):
            S[:, i] = np.where(rng.random(len(S)) < expit(2 * (Hu[:, i] + S @ J[:, i])), 1.0, -1.0)
    return (S.reshape(R, T, N) > 0).astype(np.uint8)
