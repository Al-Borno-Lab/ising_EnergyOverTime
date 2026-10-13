"""E12 DEC signal: cross-validated ridge decoder of the peak-acceleration time from single-neuron rates (see PREREGISTRATION section 4)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
T, SIGMA, WIN = 700, 3, (-50, 100)

def sm(a, sigma=SIGMA, axis=0):
    return gaussian_filter1d(np.asarray(a, float), sigma, axis=axis, mode="nearest")

def _ridge_path(Xtr, ytr, alphas):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
    Xs = (Xtr - mu) / sd
    ym = ytr.mean()
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Uty = U.T @ (ytr - ym)
    W = [Vt.T @ (S / (S ** 2 + a) * Uty) for a in alphas]
    return mu, sd, ym, W


def decode_condition(X, fold, pa, alphas=np.logspace(-2, 4, 13), sigma=SIGMA, seed=0):
    """Held-out decoder score (T,) averaged over reaches, plus held-out R^2 in the search window.
    Features: smoothed single-reach rates (T,N). Target: gaussian bump (sd 5 bins) at pa. Train on window bins only."""
    R, _, N = X.shape
    F = sm(X.astype(float), sigma, axis=1)                               # (R,T,N)
    t = np.arange(T)
    y = np.exp(-(t - pa) ** 2 / (2 * 5.0 ** 2))
    win = np.arange(pa + WIN[0], pa + WIN[1] + 1)
    pred = np.full((R, T), np.nan)
    for f in np.unique(fold):
        te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
        if len(te) == 0 or len(tr) < 3:
            continue
        # inner 3-fold CV over training reaches for alpha
        rng = np.random.RandomState(seed + int(f))
        grp = rng.permutation(len(tr)) % 3
        err = np.zeros(len(alphas))
        for g in range(3):
            a_, b_ = tr[grp != g], tr[grp == g]
            Xa = F[a_][:, win].reshape(-1, N); ya = np.tile(y[win], len(a_))
            Xb = F[b_][:, win].reshape(-1, N); yb = np.tile(y[win], len(b_))
            mu, sd, ym, W = _ridge_path(Xa, ya, alphas)
            for ai, w in enumerate(W):
                err[ai] += ((((Xb - mu) / sd) @ w + ym - yb) ** 2).mean()
        ai = int(np.argmin(err))
        Xa = F[tr][:, win].reshape(-1, N); ya = np.tile(y[win], len(tr))
        mu, sd, ym, W = _ridge_path(Xa, ya, alphas)
        w = W[ai]
        pred[te] = ((F[te] - mu) / sd) @ w + ym
    ok = ~np.isnan(pred[:, 0])
    score = pred[ok].mean(0)
    p = pred[ok][:, win]; yy = np.tile(y[win], (p.shape[0], 1))
    r2 = 1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    return score, float(r2)


