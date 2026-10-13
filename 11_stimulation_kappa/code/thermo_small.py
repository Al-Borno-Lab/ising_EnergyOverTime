"""Window thermodynamics for S2. `unit_thermo` is the entropy / multi-information / susceptibility part of
09_claude_science_handoff/code/thermo_model.py::thermo (same formulas, trimmed), copied so E11 is self-contained.
Fixed-J (kappa = 0) quantities at the fitted fields, per window unit: I = S_indep - S_Ising (bits), chiJ = sum_{i != j} Cov(s_i, s_j)."""
import numpy as np
from scipy.special import logsumexp
import stimfit_model as sm

LN2 = np.log(2.0)


def unit_thermo(H, J):
    T, N = H.shape; S = sm.states(N); q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    I = np.empty(T); chi = np.empty(T)
    for t in range(T):
        L = S @ H[t] + q; lz = logsumexp(L); P = np.exp(L - lz)
        m = S.T @ P; meanE = -(P * L).sum()
        Sent = (lz + meanE) / LN2
        pi = np.clip((1 + m) / 2, 1e-12, 1 - 1e-12)
        S_ind = -(pi * np.log2(pi) + (1 - pi) * np.log2(1 - pi)).sum()
        Css = (S * P[:, None]).T @ S - np.outer(m, m)
        I[t] = S_ind - Sent; chi[t] = Css.sum() - np.trace(Css)
    return I, chi


def window_thermo(Xs, groups, B, res):
    units = sm.build_units(Xs, groups)
    sf = sm.StimFit(units, B)
    H = sf.fields(res["W"], res["D"]); n = units["n"]; N = units["N"]
    out = {}
    for g, name in ((0, "pulse"), (1, "control")):
        sel = np.nonzero(units["ug"] == g)[0]
        I, chi = unit_thermo(H[sel], res["J"])
        w = n[sel] / n[sel].sum()
        out["I_per_N_" + name] = float((w * I).sum() / N); out["chiJ_per_N_" + name] = float((w * chi).sum() / N)
    out["dI_per_N"] = out["I_per_N_pulse"] - out["I_per_N_control"]
    out["dchiJ_per_N"] = out["chiJ_per_N_pulse"] - out["chiJ_per_N_control"]
    return out
