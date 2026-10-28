"""Unit tests for stimfit_model: likelihood vs brute force per (reach, bin), and analytic gradient vs finite differences."""
import itertools
import numpy as np
import stimfit_model as sm


def toy(N=4, T=30, Rs=(7, 6, 5), seed=0):
    rng = np.random.default_rng(seed)
    Xs = [rng.integers(0, 2, (R, T, N)).astype(np.uint8) for R in Rs]
    masks1 = np.zeros((Rs[1], T), bool)
    on = rng.integers(10, 12, Rs[1])
    for r in range(Rs[1]): masks1[r, on[r]:on[r] + 4] = True
    masks0 = np.zeros((Rs[0], T), bool); masks0[:, 11:15] = True
    groups = [dict(block=1, mask=masks1, name="pulse"), dict(block=0, mask=masks0, name="control")]
    from scipy.interpolate import BSpline
    kn = np.arange(0, T + 1, 6.0); k = 3
    t = np.concatenate([[kn[0]] * k, kn, [kn[-1]] * k]); nb = len(t) - k - 1
    x = np.arange(T, dtype=float)
    B = np.nan_to_num(np.column_stack([BSpline(t, np.eye(nb)[i], k, extrapolate=False)(x) for i in range(nb)]))
    B[-1] = B[-2] if B[-1].sum() == 0 else B[-1]
    return Xs, groups, B


def brute_ll(Xs, groups, B, sf, p):
    W, D, kap, J = sf.unpack(p)
    N = Xs[0].shape[2]; S = sm.states(N); q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    ll = 0.0
    gid = [np.full(X.shape[:2], -1) for X in Xs]
    for g, grp in enumerate(groups): gid[grp["block"]][grp["mask"]] = g
    for b, X in enumerate(Xs):
        Hb = B @ W[b]
        for r in range(X.shape[0]):
            for t in range(X.shape[1]):
                g = gid[b][r, t]; h = Hb[t] + (D[g] if g >= 0 else 0); a = 1 + (kap[g] if g >= 0 else 0)
                E = S @ h + a * q
                s = 2.0 * X[r, t] - 1
                ll += s @ h + 0.5 * a * s @ J @ s - np.logaddexp.reduce(E)
    return ll


def test_all():
    Xs, groups, B = toy()
    units = sm.build_units(Xs, groups)
    sf = sm.StimFit(units, B)
    rng = np.random.default_rng(1)
    p = rng.normal(0, 0.3, sf.size)
    _, _, ll = sf.objective(p)
    ref = brute_ll(Xs, groups, B, sf, p)
    print("loglik fast %.12f brute %.12f diff %.2e" % (ll, ref, abs(ll - ref)))
    assert abs(ll - ref) < 1e-9 * max(1, abs(ref))
    gg = sf.f(p)[1]; worst = 0.0
    for i in rng.choice(sf.size, 40, replace=False).tolist() + list(range(sf.slices["k"][0], sf.slices["k"][1])):
        e = np.zeros(sf.size); e[i] = 1e-5
        num = (sf.f(p + e)[0] - sf.f(p - e)[0]) / 2e-5
        worst = max(worst, abs(num - gg[i]) / max(1e-3, abs(gg[i])))
    print("gradient central-difference worst relative error %.2e" % worst)
    assert worst < 1e-6
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_all()
