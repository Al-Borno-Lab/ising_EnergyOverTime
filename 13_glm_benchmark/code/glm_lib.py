"""E13: coupled-GLM benchmark library. Conditional logistic models per neuron on top of the spline baseline of tvfield_model."""
import numpy as np
import tvfield_model as tv

T = 700
LAMB, LAMJ = 1e-5, 1e-3
MODELS_ALL = ["M0", "M1", "SYNC", "LAG1", "LAG1H", "SYNC_LAG1", "SYNC_H", "LAG2", "MLAG20", "SYNC_E"]
MODELS_NULL = ["M0", "SYNC", "LAG1H"]
PHASES = ["baseline", "acceleration", "deceleration", "post"]


def phase_slices(lm):
    zc = int(lm["zero_cross"])
    return {"baseline": slice(zc - 100, zc - 40), "acceleration": slice(int(lm["onset"]), zc),
            "deceleration": slice(zc, int(lm["decel_end"])), "post": slice(int(lm["decel_end"]), int(lm["decel_end"]) + 50)}


def _shift(a, k):
    """a (R,T,...) -> value at t-k, zero-padded."""
    out = np.zeros_like(a)
    out[:, k:] = a[:, :-k]
    return out


def feature_blocks(S, i, zc):
    """Blocks for neuron i. S (R,T,N) spins. Returns dict of (R,T,d) arrays."""
    N = S.shape[2]
    oth = np.delete(np.arange(N), i)
    So = S[:, :, oth]
    ir = np.zeros(T); ir[zc - 50:zc + 50] = 1
    il = np.zeros(T); il[zc + 50:zc + 150] = 1
    hist = np.stack([_shift(S[:, :, i:i + 1], k)[:, :, 0] for k in (1, 2, 3)], 2)
    return dict(sync=So, lag1=_shift(So, 1), lag2=_shift(So, 2), hist=hist,
                sync_r=So * ir[None, :, None], sync_l=So * il[None, :, None])


SPEC = {"SYNC": ["sync"], "LAG1": ["lag1"], "LAG1H": ["lag1", "hist"], "SYNC_LAG1": ["sync", "lag1"],
        "SYNC_H": ["sync", "hist"], "LAG2": ["lag2"], "SYNC_E": ["sync", "sync_r", "sync_l"]}


def cond_ll(F, y, w):
    z = F @ w
    return -(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))


def cv_models(X, fold, B, zc, models, lamj=LAMJ, lamb=LAMB):
    """5-fold CV (given folds) held-out conditional LL per bin summed over neurons: dict model -> (R,T) (nan if fold empty)."""
    R, _, N = X.shape
    S = 2.0 * X - 1
    K = B.shape[1]
    res = {m: np.full((R, T), np.nan) for m in models}
    for m in models:
        if m not in ("M1", "MLAG20"): res[m][:] = np.nan
    glm = [m for m in models if m == "M0" or m in SPEC]
    folds = [f for f in np.unique(fold) if (fold == f).any() and (fold != f).sum() >= 3]
    for f in folds:
        te = fold == f; tr = ~te
        for m in models:
            if m == "M1":
                res[m][te] = tv.pseudo_ll(X[te], tv.fit(X[tr], B, "M1"))
            elif m == "MLAG20":
                res[m][te] = tv.pseudo_ll(X[te], tv.fit(X[tr], B, "Mlag"))
        acc = {m: np.zeros((te.sum(), T)) for m in glm}
        for i in range(N):
            blk = feature_blocks(S, i, zc)
            Bt_tr = np.tile(B, (tr.sum(), 1)); Bt_te = np.tile(B, (te.sum(), 1))
            ytr = X[tr][:, :, i].reshape(-1).astype(float); yte = X[te][:, :, i].reshape(-1).astype(float)
            for m in glm:
                names = [] if m == "M0" else SPEC[m]
                Ftr = np.hstack([Bt_tr] + [blk[n][tr].reshape(tr.sum() * T, -1) for n in names])
                Fte = np.hstack([Bt_te] + [blk[n][te].reshape(te.sum() * T, -1) for n in names])
                pen = np.concatenate([np.full(K, lamb), np.full(Ftr.shape[1] - K, lamj)])
                w = tv._fit_logistic(Ftr, ytr, pen)
                acc[m] += cond_ll(Fte, yte, w).reshape(-1, T)
        for m in glm: res[m][te] = acc[m]
    return res


def sample_m1(h, J, nch, rng, sweeps=40):
    """Gibbs sample (nch, T, N) binary states from the Ising model with fields h (T,N) and couplings J, independent over time."""
    T_, N = h.shape
    s = 2.0 * (rng.rand(nch, T_, N) < 1 / (1 + np.exp(-2 * h))[None]) - 1
    for _ in range(sweeps):
        for i in range(N):
            z = 2 * (h[None, :, i] + s @ J[i])
            s[:, :, i] = np.where(rng.rand(nch, T_) < 1 / (1 + np.exp(-z)), 1.0, -1.0)
    return ((s + 1) / 2).astype(np.uint8)
