"""Shared functions for E12: signals, extremum search, decoder. See PREREGISTRATION_2026-10-12.md (+ amendment A1)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
atanh = np.arctanh
import tvfield_model as tv
from decoder import decode_condition

T = 700
SIGMA = 3
SHIFTS = (0, -150, -200, -250)          # 0 = real window; others = rest controls (-1.5/-2/-2.5 s)
WIN = (-50, 100)                        # search window relative to peak acceleration
HIT = (-20, 0)                          # hit window relative to peak velocity (zc)
SIGNALS = ["FT", "FT0", "FR", "dFR", "POLb", "TRAJ", "DEC"]
DIRECTION = {"FT": "min", "FT0": "min", "FR": "min", "dFR": "max", "POLb": "min", "TRAJ": "max", "TRAJd": "max", "DEC": "max", "FTu": "min", "FT0u": "min"}


def sm(a, sigma=SIGMA, axis=0):
    return gaussian_filter1d(np.asarray(a, float), sigma, axis=axis, mode="nearest")


def field_term(X, h, sigma=SIGMA):
    """-sum_i h_i(t) * mean_r s_i(r,t), smoothed. X (R,T,N), h (T,N)."""
    sbar = (2.0 * X - 1).mean(0)
    return sm(-(h * sbar).sum(1), sigma)


def pca_speed(P, zc, k=3):
    """P: smoothed PSTH (T,N). Cells z-scored by baseline; speed of top-k PC trajectory."""
    b = slice(zc - 100, zc - 40)
    Z = (P - P[b].mean(0)) / np.maximum(P[b].std(0), 1e-6)
    Zc = Z - Z.mean(0)
    k = min(k, Z.shape[1])
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    PC = Zc @ Vt[:k].T
    speed = np.linalg.norm(np.gradient(PC, axis=0), axis=1)
    dist = np.linalg.norm(PC - PC[b].mean(0), axis=1)     # distance from the baseline centroid (sensitivity)
    return speed, dist


def pol_signals(X, zc, sigma=SIGMA):
    P = sm(X.mean(0), sigma)                         # (T,N) smoothed PSTH
    b = slice(zc - 100, zc - 40)
    pbase = P[b].mean(0)
    dfr = np.abs(P - pbase).mean(1)
    m = 2 * np.clip(P, 0.01, 0.99) - 1
    polb = -(m * atanh(m)).sum(1)
    sp, ds = pca_speed(P, zc)
    return dict(dFR=dfr, POLb=polb, TRAJ=sp, TRAJd=ds)


def raw_signals(X, zc, h1, h0, sigma=SIGMA):
    """Unnormalised signals dict (T,) for one condition. h1/h0: fields of M1/M0 or None."""
    out = dict(FR=sm(X.mean((0, 2)), sigma))
    out.update(pol_signals(X, zc, sigma))
    if h1 is not None: out["FT"] = field_term(X, h1, sigma)
    if h0 is not None: out["FT0"] = field_term(X, h0, sigma)
    return out


def zscore(sig, zc):
    b = slice(zc - 100, zc - 40)
    return (sig - sig[b].mean()) / max(sig[b].std(), 1e-9)


def extremum(z, direction, lo, hi):
    seg = z[lo:hi + 1]
    i = int(np.argmin(seg) if direction == "min" else np.argmax(seg))
    return lo + i, float(seg[i])


def search_rows(name, z, pa, zc, direction=None, depth=True):
    """Run the pre-registered search in the real window and the three control windows. Returns list of dict rows."""
    d = direction or DIRECTION[name]
    rows = []
    for sh in SHIFTS:
        lo, hi = pa + WIN[0] + sh, pa + WIN[1] + sh
        if lo < 0 or hi >= T:
            continue
        b, v = extremum(z, d, lo, hi)
        zcs, pas = zc + sh, pa + sh
        rows.append(dict(signal=name, shift=sh, ext_bin=b, ext_z=v, depth=abs(v) if depth else np.nan,
                         hit=int(zcs + HIT[0] <= b <= zcs + HIT[1]), hit30=int(zcs - 30 <= b <= zcs),
                         dev=abs(b - pas), rel_pa=b - pas))
    return rows


# ---------------------------------------------------------------- one condition
def fit_fields(X, zc, uniform=False):
    B = tv.bspline_basis(zc, fine=(-zc, T - zc)) if uniform else tv.bspline_basis(zc)   # uniform = 5-bin knots everywhere (A2)
    h1 = tv.fit(X, B, "M1")["h"]
    h0 = tv.fit(X, B, "M0")["h"]
    return h1, h0


def process_condition(X, fold, pa, zc, h1=None, h0=None, with_dec=True, sigma=SIGMA):
    """X (R,T,N) kept reaches. Returns (rows, extras)."""
    if h1 is None or h0 is None:
        h1, h0 = fit_fields(X, zc)
    raw = raw_signals(X, zc, h1, h0, sigma)
    h1u, h0u = fit_fields(X, zc, uniform=True)
    raw["FTu"] = field_term(X, h1u, sigma); raw["FT0u"] = field_term(X, h0u, sigma)
    rows = []
    for name, s in raw.items():
        rows += search_rows(name, zscore(s, zc), pa, zc)
    # FR_abs sensitivity: extremum of larger |z| in either direction (within each window)
    zfr = zscore(raw["FR"], zc)
    for sh in SHIFTS:
        lo, hi = pa + WIN[0] + sh, pa + WIN[1] + sh
        if lo < 0 or hi >= T: continue
        seg = zfr[lo:hi + 1]; i = int(np.argmax(np.abs(seg))); b = lo + i
        rows.append(dict(signal="FR_abs", shift=sh, ext_bin=b, ext_z=float(seg[i]), depth=abs(float(seg[i])),
                         hit=int(zc + sh - 20 <= b <= zc + sh), hit30=int(zc + sh - 30 <= b <= zc + sh),
                         dev=abs(b - pa - sh), rel_pa=b - pa - sh))
    extras = dict(traces={k: zscore(v, zc) for k, v in raw.items()})
    if with_dec:
        score, r2 = decode_condition(X, fold, pa)
        for sh in SHIFTS:
            lo, hi = pa + WIN[0] + sh, pa + WIN[1] + sh
            if lo < 0 or hi >= T: continue
            b, v = extremum(score, "max", lo, hi)
            rows.append(dict(signal="DEC", shift=sh, ext_bin=b, ext_z=np.nan, depth=np.nan,
                             hit=int(zc + sh - 20 <= b <= zc + sh), hit30=int(zc + sh - 30 <= b <= zc + sh),
                             dev=abs(b - pa - sh), rel_pa=b - pa - sh))
        extras["dec_r2"] = r2; extras["traces"]["DEC"] = score
    return rows, extras
