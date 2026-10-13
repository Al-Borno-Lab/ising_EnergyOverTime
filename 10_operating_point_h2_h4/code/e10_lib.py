"""E10 step 2 library: per-condition analysis for H2-H4 (pre-registration PREREGISTRATION_OPERATING_POINT.md, amendments 1-6).
Reuses 09_claude_science_handoff/code (thermo_model, tvfield_model) by path."""
import os, sys, time, zlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "09_claude_science_handoff", "code"))
import thermo_model as tm, tvfield_model as tv

T = 700
B = tv.bspline_basis(400)
LO, HI = 300, 420                      # thermo is evaluated only on rows 300:420 (baseline 300:360, reach 380:420)
BASE = slice(0, 60); REACH = slice(80, 120)      # relative to LO
EM = np.zeros(T, bool); EM[380:420] = True


def fit(X, W0=None, J0=None):
    if W0 is None:
        P = tv.fit(X, B, "M1"); W0 = np.linalg.lstsq(B, P["h"], rcond=None)[0]; J0 = P["J"]
    return tm.fit_exact(X, B, W0=W0, J0=J0)


def fit_kappa(X, W0, J0, lamB=1e-5, lamJ=1e-4, maxiter=500):
    """Exact ML with J_eff = (1+kappa) J in reach bins; W, J, kappa free."""
    R, _, N = X.shape; K = B.shape[1]; S = tm.states(N); iu = np.triu_indices(N, 1); nP = len(iu[0])
    s = 2.0 * X - 1; m_data = s.mean(0)
    C_out = np.einsum("rti,rtj->ij", s[:, ~EM], s[:, ~EM]) / R
    C_in = np.einsum("rti,rtj->ij", s[:, EM], s[:, EM]) / R
    nb = R * T
    def sym(v):
        M = np.zeros((N, N)); M[iu] = v; return M + M.T
    def f(p):
        W = p[:K * N].reshape(K, N); pj = p[K * N:K * N + nP]; J = sym(pj); k = p[-1]; H = B @ W
        lzo, mo, po = tm.model_moments(H[~EM], J, S); lzi, mi, pi_ = tm.model_moments(H[EM], (1 + k) * J, S)
        m = np.empty((T, N)); m[~EM] = mo; m[EM] = mi
        ll = R * (np.sum(H * m_data) + 0.5 * np.sum(J * C_out) + 0.5 * (1 + k) * np.sum(J * C_in) - lzo.sum() - lzi.sum())
        gW = B.T @ (R * (m_data - m)); gJ = R * ((C_out - po) + (1 + k) * (C_in - pi_))[iu]
        gk = R * 0.5 * np.sum(J * (C_in - pi_))
        obj = -ll / nb + lamB * np.sum(W * W) + lamJ * np.sum(pj ** 2)
        g = np.concatenate([-gW.ravel() / nb + 2 * lamB * W.ravel(), -gJ / nb + 2 * lamJ * pj, [-gk / nb]])
        return obj, g
    res = minimize(f, np.concatenate([W0.ravel(), J0[iu], [0.0]]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7))
    W = res.x[:K * N].reshape(K, N)
    return dict(h=B @ W, J=sym(res.x[K * N:K * N + nP]), W=W, kappa=float(res.x[-1]), converged=bool(res.success))


def window_thermo(F, S=None):
    return tm.thermo(F["h"][LO:HI], F["J"], S=S)


def win_delta(x): return x[REACH].mean(0) - x[BASE].mean(0)


def neuron_rho(th):
    p = (1 + th["m"]) / 2
    dd = np.abs(p[REACH].mean(0) - 0.5) - np.abs(p[BASE].mean(0) - 0.5)
    dmi = th["MI_i"][REACH].mean(0) - th["MI_i"][BASE].mean(0)
    return float(spearmanr(dd, dmi).statistic) if len(dd) > 2 else np.nan


def summarize(th, N):
    return dict(dI=win_delta(th["I"]) / N, dchiJ=win_delta(th["chiJ"]) / N, dC=win_delta(th["C"]) / N,
                dv=win_delta(th["v"]), I_base=th["I"][BASE].mean() / N, v_base=th["v"][BASE].mean(), rho=neuron_rho(th))


def analyze(Z, key, sub, nshuf=20, S_cache=None, log=print):
    """One task = one condition (sub = None, all neurons) or one 16-neuron subset (sub = (index, seed))."""
    t0 = time.time()
    X = Z["X_" + key]; keep = Z["exclude_" + key] == 0; X = X[keep]
    N_full = X.shape[2]
    if sub is not None:
        rng = np.random.default_rng(sub[1]); idx = np.sort(rng.choice(N_full, 16, replace=False)); X = X[:, :, idx]
    R, _, N = X.shape
    rng = np.random.default_rng(zlib.crc32(f"{key}-{sub}".encode()))
    rec = dict(key=key, sub=-1 if sub is None else sub[0], N_full=N_full, N=N, R=R, R_all=len(keep))
    F = fit(X); rec["converged"] = F["converged"]
    S = tm.states(N); th = window_thermo(F, S); rec.update({"real_" + k: v for k, v in summarize(th, N).items()})
    # empirical quantities (data)
    tc = tm.timelocked_cov(X); rec["emp_dcov"] = (tc[380:420].mean() - tc[300:360].mean()) / N
    p = X.mean(0); v = (p * (1 - p)).mean(1); rec["emp_dv"] = v[380:420].mean() - v[300:360].mean()
    # trial-shuffle null (each neuron's reach order permuted independently), refit
    nul = []
    for b in range(nshuf):
        Fs = fit(tv.trial_shuffle(X, rng)); ths = window_thermo(Fs, S); sm = summarize(ths, N)
        nul.append((sm["dI"], sm["rho"], sm["dchiJ"]))
    nul = np.array(nul); rec["null_dI_mean"], rec["null_dI_p95"], rec["null_dI_max"] = nul[:, 0].mean(), np.percentile(nul[:, 0], 95), nul[:, 0].max()
    rec["null_rho_mean"] = float(np.nanmean(nul[:, 1])); rec["null_dchiJ_mean"] = nul[:, 2].mean()
    # kappa (coupling scale in reach window), all reaches, then 5-fold held-out gain; fold-wise fixed-J predictions
    K1 = fit_kappa(X, F["W"], F["J"]); rec["kappa_hat"] = K1["kappa"]; rec["kappa_converged"] = K1["converged"]
    fold = np.arange(R) % 5; gain = np.zeros((R, T)); pred = []
    for f in range(5):
        tr, te = fold != f, fold == f
        if tr.sum() < 2 or te.sum() < 1: continue
        F0 = fit(X[tr]); Kf = fit_kappa(X[tr], F0["W"], F0["J"])
        gain[te] = tm.loglik_bins(X[te], Kf["h"], Kf["J"], Kf["kappa"] * Kf["J"], EM) - tm.loglik_bins(X[te], F0["h"], F0["J"])
        pred.append(summarize(window_thermo(F0, S), N)["dchiJ"])
    rec["kappa_gain_reach"] = float(gain[:, EM].mean()); rec["pred_dchiJ_folds"] = float(np.mean(pred)) if pred else np.nan
    rec["secs"] = time.time() - t0
    return rec


# ---- split version of analyze() for the contingency resubmission (identical seeds, subsets and statistics) ----
def _prep(Z, key, sub, nshuf):
    X = Z["X_" + key]; keep = Z["exclude_" + key] == 0; X = X[keep]
    N_full = X.shape[2]
    if sub is not None:
        rng = np.random.default_rng(sub[1]); idx = np.sort(rng.choice(N_full, 16, replace=False)); X = X[:, :, idx]
    rng = np.random.default_rng(zlib.crc32(f"{key}-{sub}".encode()))
    shufs = [tv.trial_shuffle(X, rng) for _ in range(nshuf)]        # same draw order as in analyze()
    return X, N_full, len(keep), shufs


def analyze_part(Z, key, sub, part, nshuf=20, nchunk=4):
    """part = 'core' (real fit, empirical, kappa, CV) or an int chunk c in range(nchunk) (shuffles c::nchunk)."""
    t0 = time.time(); X, N_full, R_all, shufs = _prep(Z, key, sub, nshuf); R, _, N = X.shape; S = tm.states(N)
    rec = dict(key=key, sub=-1 if sub is None else sub[0], N_full=N_full, N=N, R=R, R_all=R_all, part=part)
    if part == "core":
        F = fit(X); rec["converged"] = F["converged"]; th = window_thermo(F, S)
        rec.update({"real_" + k: v for k, v in summarize(th, N).items()})
        tc = tm.timelocked_cov(X); rec["emp_dcov"] = (tc[380:420].mean() - tc[300:360].mean()) / N
        p = X.mean(0); v = (p * (1 - p)).mean(1); rec["emp_dv"] = v[380:420].mean() - v[300:360].mean()
        K1 = fit_kappa(X, F["W"], F["J"]); rec["kappa_hat"] = K1["kappa"]; rec["kappa_converged"] = K1["converged"]
        fold = np.arange(R) % 5; gain = np.zeros((R, T)); pred = []
        for f in range(5):
            tr, te = fold != f, fold == f
            if tr.sum() < 2 or te.sum() < 1: continue
            F0 = fit(X[tr]); Kf = fit_kappa(X[tr], F0["W"], F0["J"])
            gain[te] = tm.loglik_bins(X[te], Kf["h"], Kf["J"], Kf["kappa"] * Kf["J"], EM) - tm.loglik_bins(X[te], F0["h"], F0["J"])
            pred.append(summarize(window_thermo(F0, S), N)["dchiJ"])
        rec["kappa_gain_reach"] = float(gain[:, EM].mean()); rec["pred_dchiJ_folds"] = float(np.mean(pred)) if pred else np.nan
    else:
        nul = []
        for b in range(part, nshuf, nchunk):
            Fs = fit(shufs[b]); sm = summarize(window_thermo(Fs, S), N); nul.append((b, sm["dI"], sm["rho"], sm["dchiJ"]))
        rec["null"] = nul
    rec["secs"] = time.time() - t0
    return rec


def merge_parts(parts):
    core = [p for p in parts if p["part"] == "core"][0]
    nul = np.array(sorted([x for p in parts if p["part"] != "core" for x in p["null"]]))[:, 1:]
    rec = {k: v for k, v in core.items() if k != "part"}
    rec["null_dI_mean"], rec["null_dI_p95"], rec["null_dI_max"] = nul[:, 0].mean(), np.percentile(nul[:, 0], 95), nul[:, 0].max()
    rec["null_rho_mean"] = float(np.nanmean(nul[:, 1])); rec["null_dchiJ_mean"] = nul[:, 2].mean()
    rec["secs"] = sum(p["secs"] for p in parts)
    return rec
