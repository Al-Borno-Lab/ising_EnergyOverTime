"""Joint stimulation model, one fit per session across blocks: exact maximum likelihood by enumeration (N <= ~20).

For block b, reach r, bin t:   P(s) ∝ exp( h_b(t)·s + ½ a_{r,t} s^T J s ),   s ∈ {-1,+1}^N
  h_b(t)  = B(t) W_b  (+ d_g if bin (r,t) is in window group g)     block-specific spline fields + window intercepts
  a_{r,t} = 1 + kappa_g if (r,t) is in window group g, else 1        J_eff = (1 + kappa) J, J shared by all blocks

A window group g = (block, boolean mask (R_b, T), name). Groups must be disjoint within a block.
Bins with the same (block, t, group) share parameters, so the likelihood is evaluated on "units" with sufficient statistics.
Extends 09_claude_science_handoff/code/thermo_model.py (same state enumeration and L2 conventions).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

T_BINS = 700


def states(N):
    return (2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1).astype(np.float64)


def build_units(Xs, groups):
    """Xs: list of (R_b, T, N) uint8 per block. groups: list of dict(block=int, mask=(R_b,T) bool, name=str).
    Returns dict of unit arrays: ub, ut, ug (-1 = no group), n, m1 (U,N), C (U,N,N)."""
    nb = len(Xs); N = Xs[0].shape[2]; T = Xs[0].shape[1]
    gid = [np.full(X.shape[:2], -1, int) for X in Xs]
    for g, grp in enumerate(groups):
        sel = grp["mask"]
        assert (gid[grp["block"]][sel] == -1).all(), "window groups overlap within a block"
        gid[grp["block"]][sel] = g
    ub, ut, ug, n, m1, C = [], [], [], [], [], []
    for b, X in enumerate(Xs):
        s = 2.0 * X - 1
        for g in np.unique(gid[b]):
            sel = gid[b] == g                                # (R,T)
            for t in np.nonzero(sel.any(0))[0]:
                rows = sel[:, t]
                st = s[rows, t]                              # (n,N)
                ub.append(b); ut.append(t); ug.append(g); n.append(len(st))
                m1.append(st.sum(0)); C.append(st.T @ st)
    return dict(ub=np.array(ub), ut=np.array(ut), ug=np.array(ug), n=np.array(n, float),
                m1=np.array(m1), C=np.array(C), nblocks=nb, N=N, T=T, G=len(groups))


class StimFit:
    """Negative penalised log-likelihood and exact gradient for the joint model."""

    def __init__(self, units, B, lamB=1e-5, lamJ=1e-4, lamD=1e-5, lamK=0.0, fit_kappa=True, chunk_budget=3e7):
        self.u = units; self.B = B; self.K = B.shape[1]; self.N = units["N"]; self.G = units["G"]
        self.nb = units["nblocks"]; self.lam = (lamB, lamJ, lamD, lamK); self.fit_kappa = fit_kappa
        self.S = states(self.N); self.iu = np.triu_indices(self.N, 1); self.nP = len(self.iu[0])
        U = len(units["n"]); self.U = U
        self.step = max(1, int(chunk_budget // len(self.S)))
        self.onehot = np.zeros((U, 1 + self.G)); self.onehot[np.arange(U), units["ug"] + 1] = units["n"]
        self.Ccls = np.zeros((1 + self.G, self.N, self.N))
        for c in range(1 + self.G):
            self.Ccls[c] = units["C"][units["ug"] + 1 == c].sum(0)
        self.nbins = units["n"].sum()
        self.slices = dict(W=(0, self.nb * self.K * self.N))
        a = self.slices["W"][1]; self.slices["D"] = (a, a + self.G * self.N)
        a = self.slices["D"][1]; self.slices["k"] = (a, a + self.G); a = self.slices["k"][1]
        self.slices["J"] = (a, a + self.nP); self.size = a + self.nP

    def unpack(self, p):
        sl = self.slices
        W = p[slice(*sl["W"])].reshape(self.nb, self.K, self.N)
        D = p[slice(*sl["D"])].reshape(self.G, self.N)
        kap = p[slice(*sl["k"])] if self.fit_kappa else np.zeros(self.G)
        J = np.zeros((self.N, self.N)); J[self.iu] = p[slice(*sl["J"])]
        return W, D, kap, J + J.T

    def fields(self, W, D):
        u = self.u
        Hb = np.stack([self.B @ W[b] for b in range(self.nb)])          # (nb,T,N)
        H = Hb[u["ub"], u["ut"]].copy()
        has = u["ug"] >= 0
        H[has] += D[u["ug"][has]]
        return H

    def model_stats(self, H, kap, J):
        """Per-unit logZ (U,), mean m (U,N), class-pooled pair sums (1+G,N,N) = sum_u n_u <ss^T>_u."""
        u = self.u; S = self.S; U = self.U
        a = np.ones(U); has = u["ug"] >= 0; a[has] = 1.0 + kap[u["ug"][has]]
        q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
        logZ = np.empty(U); m = np.empty((U, self.N)); wcls = np.zeros((len(S), 1 + self.G))
        step = max(1, int(self.step))
        for a0 in range(0, U, step):
            sl = slice(a0, min(a0 + step, U))
            L = S @ H[sl].T + q[:, None] * a[sl][None]
            Lmax = L.max(0); P = np.exp(L - Lmax); Z = P.sum(0); P /= Z; lz = np.log(Z) + Lmax     # one exp per entry
            logZ[sl] = lz; m[sl] = (S.T @ P).T; wcls += P @ self.onehot[sl]
        pair = np.stack([(S * wcls[:, c:c + 1]).T @ S for c in range(1 + self.G)])
        return logZ, m, pair

    def objective(self, p):
        u = self.u; lamB, lamJ, lamD, lamK = self.lam
        W, D, kap, J = self.unpack(p)
        H = self.fields(W, D)
        logZ, m, pair = self.model_stats(H, kap, J)
        a_c = np.concatenate([[1.0], 1.0 + kap])                         # per class
        ll = np.sum(H * u["m1"]) - np.sum(u["n"] * logZ) + 0.5 * sum(a_c[c] * np.sum(J * self.Ccls[c]) for c in range(1 + self.G))
        gH = u["m1"] - u["n"][:, None] * m                               # (U,N)
        gW = np.zeros((self.nb, self.K, self.N))
        for b in range(self.nb):
            GH = np.zeros((self.B.shape[0], self.N)); sel = u["ub"] == b
            np.add.at(GH, u["ut"][sel], gH[sel]); gW[b] = self.B.T @ GH
        gD = np.zeros((self.G, self.N)); has = u["ug"] >= 0
        np.add.at(gD, u["ug"][has], gH[has])
        diff = self.Ccls - pair                                          # (1+G,N,N)
        gJ = sum(a_c[c] * diff[c] for c in range(1 + self.G))[self.iu]
        gK = np.array([0.5 * np.sum(J * diff[1 + g]) for g in range(self.G)])   # sum_{i<j} J_ij diff_ij = ½ sum_ij
        nb_ = self.nbins
        pj = p[slice(*self.slices["J"])]
        obj = -ll / nb_ + lamB * np.sum(W * W) + lamJ * np.sum(pj ** 2) + lamD * np.sum(D * D) + lamK * np.sum(kap * kap)
        g = np.concatenate([-gW.ravel() / nb_ + 2 * lamB * W.ravel(), -gD.ravel() / nb_ + 2 * lamD * D.ravel(),
                            (-gK / nb_ + 2 * lamK * kap) if self.fit_kappa else np.zeros(self.G), -gJ / nb_ + 2 * lamJ * pj])
        return obj, g, ll

    def f(self, p):
        o, g, _ = self.objective(p)
        return o, g

    def init(self):
        u = self.u; W0 = np.zeros((self.nb, self.K, self.N))
        for b in range(self.nb):
            sel = (u["ub"] == b) & (u["ug"] < 0)
            T = self.B.shape[0]; mean = np.zeros((T, self.N)); cnt = np.zeros(T)
            np.add.at(mean, u["ut"][sel], u["m1"][sel]); np.add.at(cnt, u["ut"][sel], u["n"][sel])
            ok = cnt > 0; mt = np.clip(mean[ok] / cnt[ok][:, None], -0.95, 0.95)
            W0[b] = np.linalg.lstsq(self.B[ok], np.arctanh(mt), rcond=None)[0]
        p = np.zeros(self.size); p[slice(*self.slices["W"])] = W0.ravel()
        return p

    def fit(self, p0=None, maxiter=800, tol=1e-7):
        p0 = self.init() if p0 is None else p0
        res = minimize(self.f, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=tol, gtol=1e-7))
        W, D, kap, J = self.unpack(res.x)
        return dict(p=res.x, W=W, D=D, kappa=kap, J=J, nobj=res.fun, converged=bool(res.success), nit=res.nit,
                    ll=-res.fun * self.nbins)


def fit_session(Xs, groups, B, fit_kappa=True, p0=None, maxiter=800, tol=1e-7, **lam):
    """Convenience: build units, fit, return result dict (adds group names)."""
    units = build_units(Xs, groups)
    sf = StimFit(units, B, fit_kappa=fit_kappa, **lam)
    res = sf.fit(p0=p0, maxiter=maxiter, tol=tol)
    res["groups"] = [g["name"] for g in groups]
    return res


def wald_se(Xs, groups, B, res, idx_kappa, eps=1e-4, **lam):
    """Observed-information standard error for kappa_g by profile curvature: refit others, finite-difference kappa."""
    units = build_units(Xs, groups)
    sf = StimFit(units, B, **lam)
    k0 = res["kappa"][idx_kappa]
    def prof(k):
        p = res["p"].copy(); sk = sf.slices["k"][0] + idx_kappa; p[sk] = k
        free = np.ones(sf.size, bool); free[sk] = False
        def f(x):
            q = p.copy(); q[free] = x; o, g = sf.f(q); return o, g[free]
        r = minimize(f, p[free], jac=True, method="L-BFGS-B", options=dict(maxiter=200, ftol=1e-10, gtol=1e-8))
        return r.fun * sf.nbins
    h = 0.1
    f0, fp, fm = prof(k0), prof(k0 + h), prof(k0 - h)
    curv = (fp + fm - 2 * f0) / h ** 2
    return float(1 / np.sqrt(max(curv, 1e-9)))
