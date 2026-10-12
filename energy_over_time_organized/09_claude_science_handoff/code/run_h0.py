"""H0 synthetic validation for the operating-point test (PREREGISTRATION_OPERATING_POINT.md, section 7).
SYNTHETIC DATA ONLY - ground truth J taken from real fitted conditions; nothing here is a real-data result.
Usage: python run_h0.py NCPU"""
import sys, os, time, pickle, numpy as np, pandas as pd, multiprocessing as mp
sys.path.append(os.getcwd())
import thermo_model as tm, tvfield_model as tv
from scipy.stats import spearmanr

T = 700; BASE = slice(300, 360); REACH = slice(380, 420)
EM = np.zeros(T, bool); EM[REACH] = True
TJ = np.load("h0_truth_J.npz"); JS = {11: TJ["J11"], 15: TJ["J15"]}
B = tv.bspline_basis(400)
tt = np.arange(T)
g = np.where(tt < 395, np.exp(-0.5 * ((tt - 395) / 12) ** 2), np.exp(-0.5 * ((tt - 395) / 40) ** 2))


def truth(scen, N, rng):
    J = JS[N].copy()
    if scen in ("A", "C", "D"):
        pb = rng.uniform(0.58, 0.68, N); pr = np.full(N, 0.45) if scen != "C" else pb
    elif scen == "B":
        pb = rng.uniform(0.55, 0.65, N); pr = np.full(N, 0.80)
    elif scen == "E":
        lo = rng.random(N) < 0.3
        pb = np.where(lo, rng.uniform(0.35, 0.45, N), rng.uniform(0.58, 0.68, N)); pr = np.where(lo, 0.25, 0.45)
    if scen == "D": J[:] = 0
    p = pb[None] + (pr - pb)[None] * g[:, None]
    H = tm.fit_fields_only(2 * p - 1, J)
    Jin = 1.6 * J if scen == "C" else None
    return H, J, Jin


def win(x, sl): return x[sl].mean(0)


def summ(th, N, pre):
    o = {}
    for q in ("I", "chiJ", "C"):
        o[f"{pre}d{q}"] = (win(th[q], REACH) - win(th[q], BASE)) / N
    o[f"{pre}dv"] = win(th["v"], REACH) - win(th["v"], BASE)
    ex = th["pause_ising"] - th["pause_ind"]; o[f"{pre}dpause"] = win(ex, REACH) - win(ex, BASE)
    o[f"{pre}I_base"] = win(th["I"], BASE) / N
    return o


def thermo_piecewise(H, J, Jin):
    th = tm.thermo(H, J)
    if Jin is not None:
        ti = tm.thermo(H[EM], Jin)
        for k in th: th[k][EM] = ti[k]
    return th


def fit(X):
    P = tv.fit(X, B, "M1")
    W0 = np.linalg.lstsq(B, P["h"], rcond=None)[0]
    return tm.fit_exact(X, B, W0=W0, J0=P["J"])


def neuron_rel(th):
    p = (1 + th["m"]) / 2
    dd = np.abs(win(p, REACH) - 0.5) - np.abs(win(p, BASE) - 0.5)
    dmi = win(th["MI_i"], REACH) - win(th["MI_i"], BASE)
    return spearmanr(dd, dmi).statistic


def task(args):
    scen, N, R, rep = args; t0 = time.time()
    rng = np.random.default_rng(10000 * N + 100 * R + rep + ord(scen))
    H, J, Jin = truth(scen, N, rng)
    X = tm.sample(H, J, R, rng)
    if Jin is not None: X[:, EM] = tm.sample(H[EM], Jin, R, rng)
    th_t = thermo_piecewise(H, J, Jin); F = fit(X); th_e = tm.thermo(F["h"], F["J"])
    iu = np.triu_indices(N, 1)
    rec = dict(scen=scen, N=N, R=R, rep=rep, J_r=np.corrcoef(F["J"][iu], J[iu])[0, 1] if scen != "D" else np.nan,
               J_mae=np.abs(F["J"][iu] - J[iu]).mean(), converged=F["converged"])
    rec.update(summ(th_t, N, "true_")); rec.update(summ(th_e, N, "est_"))
    tc = tm.timelocked_cov(X); rec["emp_dcov"] = (win(tc, REACH) - win(tc, BASE)) / N
    rec["true_neuron_rho"] = neuron_rel(th_t); rec["est_neuron_rho"] = neuron_rel(th_e)
    nul = []
    for b in range(2):
        Xs = tv.trial_shuffle(X, rng); Fs = fit(Xs); ths = tm.thermo(Fs["h"], Fs["J"])
        nul.append((win(ths["I"], REACH) - win(ths["I"], BASE)) / N)
    rec["null_dI_mean"] = float(np.mean(nul)); rec["null_dI_max"] = float(np.max(nul))
    if scen in ("A", "C"):
        fold = np.arange(R) % 5; gain = np.zeros((R, T))
        for f in range(5):
            tr, te = fold != f, fold == f
            F0 = fit(X[tr]); W0 = F0["W"]
            F1 = tm.fit_exact_epoch(X[tr], B, EM, W0=W0, J0=F0["J"])
            gain[te] = tm.loglik_bins(X[te], F1["h"], F1["J"], F1["D"], EM) - tm.loglik_bins(X[te], F0["h"], F0["J"])
        rec["epochJ_gain_reach"] = gain[:, EM].mean(); rec["epochJ_gain_all"] = gain.mean()
        rec["epochJ_gain_reach_by_reach"] = gain[:, EM].mean(1).tolist()
    rec["secs"] = time.time() - t0
    return rec


if __name__ == "__main__":
    NCPU = int(sys.argv[1])
    tasks = [(s, N, R, r) for N in (15, 11) for R in (60, 20) for s in "CAEBD" for r in range(5)]
    out = []
    with mp.get_context("fork").Pool(NCPU) as pool:
        for i, rec in enumerate(pool.imap_unordered(task, tasks)):
            out.append(rec); print(i + 1, rec["scen"], rec["N"], rec["R"], rec["rep"], round(rec["secs"]), flush=True)
            if (i + 1) % 10 == 0: pickle.dump(out, open("h0_results.pkl", "wb"))
    pickle.dump(out, open("h0_results.pkl", "wb"))
    pd.DataFrame([{k: v for k, v in r.items() if k != "epochJ_gain_reach_by_reach"} for r in out]).to_csv("h0_results.csv", index=False)
    print("done")
