"""Time-varying-field pairwise model with fixed J: exact maximum likelihood (moment matching) and
time-resolved thermodynamic quantities, by exact enumeration (N <= ~20).

P_t(s) ∝ exp(h(t)·s + ½ s^T J s), s ∈ {-1,+1}^N, J symmetric with zero diagonal, h(t) = B(t) W.
Exact log-likelihood is concave in (W, J); gradients are data-minus-model moments.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit

LN2 = np.log(2.0)


def states(N):
    return (2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1).astype(np.float64)


def _chunks(T, n_states, budget=4e7):
    step = max(1, int(budget // n_states))
    return [slice(a, min(a + step, T)) for a in range(0, T, step)]


def model_moments(H, J, S=None, want=("m", "pair", "logZ")):
    """H (T,N), J (N,N). Returns per-bin logZ (T,), m (T,N), and pooled sum_t <ss^T> (N,N)."""
    T, N = H.shape
    S = states(N) if S is None else S
    q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    logZ = np.empty(T); m = np.empty((T, N)); psum = np.zeros(len(S))
    for sl in _chunks(T, len(S)):
        L = S @ H[sl].T + q[:, None]                       # (2^N, t)
        lz = logsumexp(L, axis=0); P = np.exp(L - lz)
        logZ[sl] = lz; m[sl] = (S.T @ P).T; psum += P.sum(1)
    pair = (S * psum[:, None]).T @ S if "pair" in want else None
    return logZ, m, pair


def fit_exact(X, B, W0=None, J0=None, lamB=1e-5, lamJ=1e-4, maxiter=500, tol=1e-9):
    """Exact ML. X (R,T,N) in {0,1}. B (T,K). Returns dict(h=(T,N), J, W, nll, converged)."""
    R, T, N = X.shape; K = B.shape[1]; S = states(N); iu = np.triu_indices(N, 1)
    s = 2.0 * X - 1
    m_data = s.mean(0)                                     # (T,N)
    C_data = np.einsum("rti,rtj->ij", s, s) / R            # sum_t mean_r s s^T
    nb = R * T
    def unpack(p):
        W = p[:K * N].reshape(K, N); J = np.zeros((N, N)); J[iu] = p[K * N:]; return W, J + J.T
    def f(p):
        W, J = unpack(p); H = B @ W
        logZ, m, pair = model_moments(H, J, S)
        ll = R * (np.sum(H * m_data) + 0.5 * np.sum(J * C_data) - logZ.sum())
        gH = R * (m_data - m); gW = B.T @ gH
        gJ = R * (C_data - pair)[iu]
        obj = -ll / nb + lamB * np.sum(W * W) + lamJ * np.sum(p[K * N:] ** 2)
        g = np.concatenate([-gW.ravel() / nb + 2 * lamB * W.ravel(), -gJ / nb + 2 * lamJ * p[K * N:]])
        return obj, g
    W0 = np.zeros((K, N)) if W0 is None else W0
    j0 = np.zeros(len(iu[0])) if J0 is None else J0[iu]
    res = minimize(f, np.concatenate([W0.ravel(), j0]), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=tol, gtol=1e-7))
    W, J = unpack(res.x)
    return dict(h=B @ W, J=J, W=W, nll=res.fun, converged=bool(res.success), nit=res.nit)


def fit_exact_epoch(X, B, emask, W0=None, J0=None, lamB=1e-5, lamJ=1e-4, lamD=1e-3, maxiter=500):
    """Exact ML with J_eff(t) = J + D for bins where emask (T,) is True, J elsewhere."""
    R, T, N = X.shape; K = B.shape[1]; S = states(N); iu = np.triu_indices(N, 1); nP = len(iu[0])
    s = 2.0 * X - 1; m_data = s.mean(0)
    C_out = np.einsum("rti,rtj->ij", s[:, ~emask], s[:, ~emask]) / R
    C_in = np.einsum("rti,rtj->ij", s[:, emask], s[:, emask]) / R
    nb = R * T
    def sym(v):
        M = np.zeros((N, N)); M[iu] = v; return M + M.T
    def f(p):
        W = p[:K * N].reshape(K, N); J = sym(p[K * N:K * N + nP]); D = sym(p[K * N + nP:]); H = B @ W
        lzo, mo, po = model_moments(H[~emask], J, S); lzi, mi, pi_ = model_moments(H[emask], J + D, S)
        m = np.empty((T, N)); m[~emask] = mo; m[emask] = mi
        ll = R * (np.sum(H * m_data) + 0.5 * np.sum(J * (C_out + C_in)) + 0.5 * np.sum(D * C_in) - lzo.sum() - lzi.sum())
        gW = B.T @ (R * (m_data - m)); gJ = R * ((C_out + C_in) - (po + pi_))[iu]; gD = R * (C_in - pi_)[iu]
        pj, pd = p[K * N:K * N + nP], p[K * N + nP:]
        obj = -ll / nb + lamB * np.sum(W * W) + lamJ * np.sum(pj ** 2) + lamD * np.sum(pd ** 2)
        g = np.concatenate([-gW.ravel() / nb + 2 * lamB * W.ravel(), -gJ / nb + 2 * lamJ * pj, -gD / nb + 2 * lamD * pd])
        return obj, g
    W0 = np.zeros((K, N)) if W0 is None else W0; j0 = np.zeros(nP) if J0 is None else J0[iu]
    res = minimize(f, np.concatenate([W0.ravel(), j0, np.zeros(nP)]), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7))
    W = res.x[:K * N].reshape(K, N)
    return dict(h=B @ W, J=sym(res.x[K * N:K * N + nP]), D=sym(res.x[K * N + nP:]), W=W, converged=bool(res.success))


def loglik_bins(X, H, J, D=None, emask=None, S=None):
    """Exact log-likelihood per reach and bin (R,T), nats."""
    R, T, N = X.shape; S = states(N) if S is None else S; s = 2.0 * X - 1
    def part(sl_mask, Jx):
        lz, _, _ = model_moments(H[sl_mask], Jx, S, want=())
        ss = s[:, sl_mask]
        return np.einsum("rti,ti->rt", ss, H[sl_mask]) + 0.5 * np.einsum("rti,ij,rtj->rt", ss, Jx, ss) - lz[None]
    if D is None: return part(np.ones(T, bool), J)
    out = np.empty((R, T)); out[:, ~emask] = part(~emask, J); out[:, emask] = part(emask, J + D); return out


def fit_fields_only(m_target, J, h0=None, iters=200, S=None):
    """Solve h(t) so the model marginals equal m_target (T,N) for fixed J (Newton on the convex dual)."""
    T, N = m_target.shape; S = states(N) if S is None else S
    q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    H = np.arctanh(np.clip(m_target, -0.999, 0.999)) if h0 is None else h0.copy()
    for it in range(iters):
        L = S @ H.T + q[:, None]; P = np.exp(L - logsumexp(L, axis=0))
        m = (S.T @ P).T; d = m_target - m
        if np.abs(d).max() < 1e-9: break
        for t in range(T):                                 # Newton step, Hessian = covariance
            C = (S * P[:, t][:, None]).T @ S - np.outer(m[t], m[t])
            H[t] += np.linalg.solve(C + 1e-9 * np.eye(N), d[t])
    return H


def thermo(H, J, S=None, k_pause=None):
    """Per-bin quantities. Returns dict of (T,) arrays and per-neuron (T,N) arrays. Entropies in bits."""
    T, N = H.shape; S = states(N) if S is None else S
    q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    nsil = (S < 0).sum(1)                                  # silent count per state
    k = int(np.ceil(N / 2)) if k_pause is None else k_pause
    out = {n: np.empty(T) for n in ("S", "S_ind", "I", "chiJ", "C", "v", "pause_ising", "pause_ind")}
    out["m"] = np.empty((T, N)); out["MI_i"] = np.empty((T, N))
    loc = S @ J                                            # (2^N, N): sum_j J_ij s_j
    for sl in _chunks(T, len(S) * N, budget=2e7):
        L = S @ H[sl].T + q[:, None]; lz = logsumexp(L, axis=0); P = np.exp(L - lz)        # (2^N,t)
        m = (S.T @ P).T                                    # (t,N)
        Eneg = L                                           # = -E
        meanE = -(P * Eneg).sum(0); varE = (P * Eneg ** 2).sum(0) - meanE ** 2
        Sent = (lz + meanE) / LN2                          # S = logZ + <E>  (nats) -> bits
        pi = (1 + m) / 2
        Hb = lambda p: -(np.where(p > 0, p * np.log2(np.clip(p, 1e-300, 1)), 0) + np.where(p < 1, (1 - p) * np.log2(np.clip(1 - p, 1e-300, 1)), 0))
        S_ind = Hb(pi).sum(1)
        Css = np.einsum("st,si,sj->tij", P, S, S) - np.einsum("ti,tj->tij", m, m)
        chiJ = Css.sum((1, 2)) - np.einsum("tii->t", Css)
        # conditional entropy of s_i given the rest: E_s[H_b(sigmoid(2(h_i + loc_i)))]
        z = 2 * (H[sl][None, :, :] + loc[:, None, :])     # (2^N,t,N)
        pc = expit(z); Hc = Hb(pc)
        Hcond = np.einsum("st,stn->tn", P, Hc)
        MI = Hb(pi) - Hcond
        # synchronous pause: P(#silent >= k)
        pz = (P * (nsil >= k)[:, None]).sum(0)
        q0 = 1 - pi; pind = np.empty(len(q0))              # Poisson-binomial tail
        for a, qq in enumerate(q0):
            dist = np.array([1.0])
            for x in qq: dist = np.convolve(dist, [1 - x, x])
            pind[a] = dist[k:].sum()
        out["S"][sl] = Sent; out["S_ind"][sl] = S_ind; out["I"][sl] = S_ind - Sent; out["chiJ"][sl] = chiJ
        out["C"][sl] = varE; out["v"][sl] = (pi * (1 - pi)).mean(1); out["pause_ising"][sl] = pz; out["pause_ind"][sl] = pind
        out["m"][sl] = m; out["MI_i"][sl] = MI
    return out


def sample(H, J, R, rng, S=None):
    """Exact sampling: R independent reaches, one state per bin from P_t. Returns (R,T,N) uint8."""
    T, N = H.shape; S = states(N) if S is None else S
    q = 0.5 * np.einsum("si,ij,sj->s", S, J, S); X = np.empty((R, T, N), np.uint8)
    for sl in _chunks(T, len(S)):
        L = S @ H[sl].T + q[:, None]; P = np.exp(L - logsumexp(L, axis=0))
        cdf = np.cumsum(P, 0)
        for j, t in enumerate(range(sl.start, sl.stop)):
            idx = np.searchsorted(cdf[:, j], rng.random(R) * cdf[-1, j])
            X[:, t] = (S[np.minimum(idx, len(S) - 1)] > 0)
    return X


def timelocked_cov(X):
    """Empirical across-reach covariance summed over i != j, per bin, unbiased (R-1). X (R,T,N) -> (T,)"""
    s = 2.0 * X - 1; d = s - s.mean(0, keepdims=True); R = X.shape[0]
    C = np.einsum("rti,rtj->tij", d, d) / (R - 1)
    return C.sum((1, 2)) - np.einsum("tii->t", C)
