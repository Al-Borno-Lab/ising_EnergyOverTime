"""E13: coupled-GLM benchmark library. Conditional logistic models per neuron on top of the spline baseline of tvfield_model."""
import numpy as np
import tvfield_model as tv

T = 700
LAMB, LAMJ = 1e-5, 1e-3
MODELS_ALL = ["M0", "M1", "SYNC", "LAG1", "LAG1H", "SYNC_LAG1", "SYNC_H", "LAG2", "MLAG20", "SYNC_E"]
MODELS_NULL = ["M0", "SYNC", "LAG1H"]
PHASES = ["baseline", "acceleration", "deceleration", "post"]
# Pre-registered constants (PREREGISTRATION_2026-10-12.md sections 1-4 and amendment A1); asserted below and stored in every output.
PREREG = dict(T=700, lamb=1e-5, lamj=1e-3, n_folds=5, baseline=(-100, -40), post_len=50, reach_epoch=(-50, 50), late_epoch=(50, 150),
              hist_lags=(1, 2, 3), n_null_shuffles=20, n_ref_reps=5, ref_sweeps=40, bspline="tvfield_model.bspline_basis(zc) fine=(-60,160) fine_step=5 coarse_step=25")
assert T == PREREG["T"] and LAMB == PREREG["lamb"] and LAMJ == PREREG["lamj"]


def assert_design(X, fold, B, zc, lamj=LAMJ, lamb=LAMB, lm=None):
    """Assert the pre-registered fitted ranges, folds and basis (protocol: Review before merge)."""
    assert X.ndim == 3 and X.shape[1] == T, X.shape
    assert fold.shape == (X.shape[0],) and set(np.unique(fold)) <= set(range(PREREG["n_folds"])), "folds must be fold_k values 0..4"
    assert np.array_equal(B, tv.bspline_basis(zc)) and B.shape[0] == T
    assert lamb == PREREG["lamb"]
    if lm is not None:
        assert lm["onset"] < zc < lm["decel_end"] and lm["decel_end"] + PREREG["post_len"] <= T
        assert zc + PREREG["baseline"][0] >= 0 and zc + PREREG["late_epoch"][1] <= T


def phase_slices(lm):
    zc = int(lm["zero_cross"])
    sl = {"baseline": slice(zc + PREREG["baseline"][0], zc + PREREG["baseline"][1]), "acceleration": slice(int(lm["onset"]), zc),
          "deceleration": slice(zc, int(lm["decel_end"])), "post": slice(int(lm["decel_end"]), int(lm["decel_end"]) + PREREG["post_len"])}
    assert sl["baseline"].stop - sl["baseline"].start == 60 and sl["post"].stop - sl["post"].start == 50
    assert sl["acceleration"].stop > sl["acceleration"].start and sl["deceleration"].stop > sl["deceleration"].start
    return sl


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
    ir = np.zeros(T); ir[zc + PREREG["reach_epoch"][0]:zc + PREREG["reach_epoch"][1]] = 1
    il = np.zeros(T); il[zc + PREREG["late_epoch"][0]:zc + PREREG["late_epoch"][1]] = 1
    hist = np.stack([_shift(S[:, :, i:i + 1], k)[:, :, 0] for k in PREREG["hist_lags"]], 2)
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
    assert_design(X, fold, B, zc, lamj, lamb)
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
