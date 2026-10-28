"""bspline_basis copied verbatim from 09_claude_science_handoff/code/tvfield_model.py (git commit of E09), so E11 is self-contained."""
import numpy as np
from scipy.interpolate import BSpline
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


def bspline_basis_crop(lo=300, hi=560, fine=(340, 560), fine_step=5, coarse_step=25, k=3):
    """Cubic B-spline basis built directly on bins lo..hi-1 (clamped ends), 5-bin knots inside `fine`, 25-bin knots before it.
    Replaces cropping bspline_basis(400)[lo:hi], whose edge columns are only partly supported in the range and converge slowly."""
    kn = np.unique(np.concatenate([np.arange(lo, fine[0], coarse_step), np.arange(fine[0], min(fine[1], hi), fine_step), [hi - 1]])).astype(float)
    t = np.concatenate([[kn[0]] * k, kn, [kn[-1]] * k]); nb = len(t) - k - 1
    x = np.arange(lo, hi, dtype=float)
    B = np.column_stack([BSpline(t, np.eye(nb)[i], k, extrapolate=False)(x) for i in range(nb)])
    B = np.nan_to_num(B); B[-1] = B[-2] if B[-1].sum() == 0 else B[-1]
    return B
