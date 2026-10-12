"""E10 step 1: synthetic check of pre-registration amendments 2-3 (see amendment 5 for pass criteria).
SYNTHETIC DATA ONLY (method validation). Reuses ../09_claude_science_handoff/code by path.
Usage: python step1_synthetic_amendments.py NCPU OUTDIR TRUTH_J_NPZ"""
import sys, os, time, pickle
import numpy as np, pandas as pd, multiprocessing as mp
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
H09 = os.path.join(HERE, "..", "..", "09_claude_science_handoff", "code")
NCPU, OUT, TRUTH = int(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(TRUTH))           # run_h0 loads h0_truth_J.npz from cwd at import
sys.path.insert(0, H09)
import thermo_model as tm, tvfield_model as tv
import run_h0 as h0
from scipy.stats import spearmanr
EM, B = h0.EM, h0.B
NSHUF = 10


def fit_kappa(X, W0, J0, lamB=1e-5, lamJ=1e-4, maxiter=500):
    """Exact ML with J_eff = (1+kappa) J in reach bins (EM); W, J, kappa free."""
    R, T, N = X.shape; K = B.shape[1]; S = tm.states(N); iu = np.triu_indices(N, 1); nP = len(iu[0])
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
    x0 = np.concatenate([W0.ravel(), J0[iu], [0.0]])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7))
    W = res.x[:K * N].reshape(K, N)
    return dict(h=B @ W, J=sym(res.x[K * N:K * N + nP]), W=W, kappa=float(res.x[-1]), converged=bool(res.success))


def task(args):
    scen, N, R, rep = args; t0 = time.time()
    rng = np.random.default_rng(10000 * N + 100 * R + rep + ord(scen))
    H, J, Jin = h0.truth(scen, N, rng)
    X = tm.sample(H, J, R, rng)
    if Jin is not None: X[:, EM] = tm.sample(H[EM], Jin, R, rng)
    th_t = h0.thermo_piecewise(H, J, Jin); F = h0.fit(X); th_e = tm.thermo(F["h"], F["J"])
    rec = dict(scen=scen, N=N, R=R, rep=rep, converged=F["converged"], true_kappa=0.6 if scen == "C" else 0.0)
    rec["true_neuron_rho"] = h0.neuron_rel(th_t); rec["est_neuron_rho"] = h0.neuron_rel(th_e)
    rec["est_dI"] = (th_e["I"][h0.REACH].mean() - th_e["I"][h0.BASE].mean()) / N
    if scen in ("A", "C"):
        K1 = fit_kappa(X, F["W"], F["J"]); rec["kappa_hat"] = K1["kappa"]; rec["kappa_converged"] = K1["converged"]
        fold = np.arange(R) % 5; gain = np.zeros((R, h0.T))
        for f in range(5):
            tr, te = fold != f, fold == f
            F0 = h0.fit(X[tr]); Kf = fit_kappa(X[tr], F0["W"], F0["J"])
            gain[te] = (tm.loglik_bins(X[te], Kf["h"], Kf["J"], Kf["kappa"] * Kf["J"], EM)
                        - tm.loglik_bins(X[te], F0["h"], F0["J"]))
        rec["kappa_gain_reach"] = float(gain[:, EM].mean())
    else:
        nul = [h0.neuron_rel(tm.thermo(*(lambda Fs: (Fs["h"], Fs["J"]))(h0.fit(tv.trial_shuffle(X, rng))))) for _ in range(NSHUF)]
        rec["null_neuron_rho"] = float(np.mean(nul)); rec["null_neuron_rho_sd"] = float(np.std(nul))
        rec["rho_diff"] = rec["est_neuron_rho"] - rec["null_neuron_rho"]
    rec["secs"] = time.time() - t0
    return rec


if __name__ == "__main__":
    tasks = [(s, N, R, r) for N in (15, 11) for R in (60, 20) for s in "ACDE" for r in range(5)]
    if len(sys.argv) > 4 and sys.argv[4] == "smoke": tasks = [("C", 11, 20, 0), ("E", 11, 20, 0)]
    out = []
    with mp.get_context("fork").Pool(NCPU) as pool:
        for i, rec in enumerate(pool.imap_unordered(task, tasks)):
            out.append(rec); print(i + 1, len(tasks), rec["scen"], rec["N"], rec["R"], rec["rep"], round(rec["secs"]), flush=True)
            pickle.dump(out, open(os.path.join(OUT, "step1_results.pkl"), "wb"))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "step1_results.csv"), index=False)
    print("done")
