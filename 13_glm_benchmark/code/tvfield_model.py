# Vendored unchanged from 09_claude_science_handoff/code/tvfield_model.py (E13 must be self-contained).

"""Pairwise maximum-entropy model with time-varying fields h_i(t) and static couplings J,
fitted by L2-penalized pseudo-likelihood (one logistic regression per neuron, J symmetrized).
Spins s in {-1,+1}; P(s|t) ~ exp(sum_i h_i(t) s_i + sum_{i<j} J_ij s_i s_j).
Conditional: P(s_i=+1 | s_-i, t) = sigmoid(2 (h_i(t) + sum_j J_ij s_j)).
Models: 'M0' h(t), J=0 (time-varying independent); 'M1' h(t)+J; 'Ms' static h + J; 'Mlag' h(t) + weights on
other neurons' states LAGGED by `lag` bins (control for slow co-fluctuation; not an Ising model)."""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from scipy.special import expit, logsumexp
T = 700

def bspline_basis(zc, fine=(-60, 160), fine_step=5, coarse_step=25, k=3):
    """Cubic B-spline basis over 0..T-1, knots every `fine_step` bins inside zc+fine, `coarse_step` elsewhere."""
    a, b = zc + fine[0], zc + fine[1]
    kn = np.unique(np.concatenate([np.arange(0, a, coarse_step), np.arange(a, b, fine_step), np.arange(b, T, coarse_step), [T - 1]])).astype(float)
    t = np.concatenate([[kn[0]] * k, kn, [kn[-1]] * k])
    nb = len(t) - k - 1
    x = np.arange(T, dtype=float)
    B = np.column_stack([BSpline(t, np.eye(nb)[i], k, extrapolate=False)(x) for i in range(nb)])
    B = np.nan_to_num(B); B[-1] = B[-2] if B[-1].sum() == 0 else B[-1]
    return B

def _fit_logistic(F, y, pen, w0=None):
    """min mean logloss + sum(pen*w^2); F (n,p), y in {0,1}."""
    n, p = F.shape
    def f(w):
        z = F @ w
        l = np.logaddexp(0, z) - y * z
        g = F.T @ (expit(z) - y) / n + 2 * pen * w
        return l.mean() + (pen * w * w).sum(), g
    r = minimize(f, np.zeros(p) if w0 is None else w0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return r.x

def fit(X, B, model, lamJ=1e-3, lamB=1e-5, lag=20):
    """X: (R,T,N) uint8. B: (T,K) basis (use ones((T,1)) for static). Returns dict(h=(T,N), J=(N,N))."""
    R, _, N = X.shape
    S = 2.0 * X - 1
    Bt = np.tile(B, (R, 1))                                    # (R*T, K)
    K = B.shape[1]
    H = np.zeros((T, N)); W = np.zeros((N, N))
    for i in range(N):
        y = X[:, :, i].reshape(-1).astype(float)
        if model == "M0":
            F = Bt; pen = np.full(K, lamB)
        else:
            oth = np.delete(np.arange(N), i)
            So = S[:, :, oth]
            if model == "Mlag": So = np.concatenate([np.zeros((R, lag, N - 1)), So[:, :-lag]], 1)
            F = np.hstack([Bt, So.reshape(-1, N - 1)]); pen = np.concatenate([np.full(K, lamB), np.full(N - 1, lamJ)])
        w = _fit_logistic(F, y, pen)
        H[:, i] = B @ w[:K] / 2
        if model != "M0": W[i, np.delete(np.arange(N), i)] = w[K:] / 2
    J = (W + W.T) / 2 if model in ("M1", "Ms") else W
    if model in ("M1", "Ms"):   # refit fields with J fixed and symmetric so conditionals are consistent
        for i in range(N):
            y = X[:, :, i].reshape(-1).astype(float)
            off = (S @ J[i]).reshape(-1)                         # sum_j J_ij s_j  (J_ii=0)
            def f(wb, off=off, y=y):
                z = Bt @ wb + 2 * off
                g = Bt.T @ (expit(z) - y) / len(y) + 2 * lamB * wb
                return (np.logaddexp(0, z) - y * z).mean() + lamB * (wb * wb).sum(), g
            wb = minimize(f, 2 * np.linalg.lstsq(B, H[:, i], rcond=None)[0], jac=True, method="L-BFGS-B").x
            H[:, i] = B @ wb / 2
    return dict(h=H, J=J, model=model, lag=lag)

def pseudo_ll(X, par):
    """Per-bin held-out conditional log-likelihood summed over neurons: (R,T)."""
    S = 2.0 * X - 1; R, _, N = X.shape
    if par["model"] == "Mlag":
        L = par["lag"]; Sl = np.concatenate([np.zeros((R, L, N)), S[:, :-L]], 1)
        field = par["h"][None] + Sl @ par["J"].T
    else:
        field = par["h"][None] + S @ par["J"].T                 # J symmetric, zero diagonal
    z = 2 * field
    return (-(np.logaddexp(0, -z) * X + np.logaddexp(0, z) * (1 - X))).sum(2)

def exact_ll(X, par, chunk=2**16):
    """Exact per-bin log-likelihood (R,T) for Ising models (N <= ~20)."""
    S = 2.0 * X - 1; R, _, N = X.shape; h, J = par["h"], par["J"]
    states = 2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1   # (2^N, N)
    q = 0.5 * np.einsum("si,ij,sj->s", states, J, states)
    logZ = np.full(T, -np.inf)
    for a in range(0, 2 ** N, chunk):
        E = states[a:a + chunk] @ h.T + q[a:a + chunk, None]              # (chunk, T)
        logZ = np.logaddexp(logZ, logsumexp(E, axis=0))
    num = (S * h[None]).sum(2) + 0.5 * np.einsum("rti,ij,rtj->rt", S, J, S)
    return num - logZ[None]

def trial_shuffle(X, rng):
    """Permute reach identity independently per neuron (keeps each neuron's PSTH, removes co-variation)."""
    R, _, N = X.shape
    return np.stack([X[rng.permutation(R), :, i] for i in range(N)], 2)
