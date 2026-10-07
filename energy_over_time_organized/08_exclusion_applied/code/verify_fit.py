"""Stage 2: verify binarization, energy terms and the stored Ising fit for all 63 conditions.
(a) spikes in per_reach_state == bin10_spike counts>0, rows 100:800, same reach order
(b) recompute sum_h s, sum_J ss, E from stored h/J and compare with per_reach_state columns
(c) do stored (h, J) reproduce the data moments? exact enumeration N<=20, Gibbs sampling N>20
(d) independent pseudo-likelihood refit -> correlation with stored J, h"""
import glob, os, sys, numpy as np, pandas as pd, scipy.io as sio
from scipy.optimize import minimize
from scipy.special import expit
from multiprocessing import Pool
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
B10 = "/data001/projects/enserrog/AbigailData/energy_over_time"
b10 = {os.path.basename(p)[:6]: p for p in glob.glob(f"{B10}/*_bin10_spike.mat")}
rng0 = np.random.default_rng(0)

def moments_exact(h, J):
    N = len(h); S = 2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1
    E = S @ h + 0.5 * np.einsum("si,ij,sj->s", S, J, S) if N <= 14 else None
    if E is None:   # chunked for N 15..20
        E = np.empty(len(S))
        for a in range(0, len(S), 1 << 16): Sa = S[a:a + (1 << 16)]; E[a:a + len(Sa)] = Sa @ h + 0.5 * np.einsum("si,ij,sj->s", Sa, J, Sa)
    p = np.exp(E - E.max()); p /= p.sum()
    m = p @ S; C = np.zeros((N, N))
    for a in range(0, len(S), 1 << 16):
        Sa = S[a:a + (1 << 16)]; C += (Sa * p[a:a + (1 << 16), None]).T @ Sa
    return m, C

def moments_gibbs(h, J, chains=4000, sweeps=1500, burn=500, seed=1):
    rng = np.random.default_rng(seed); N = len(h); s = rng.choice([-1.0, 1.0], (chains, N))
    msum = np.zeros(N); Csum = np.zeros((N, N)); n = 0
    for it in range(sweeps):
        for i in rng.permutation(N):
            f = h[i] + s @ J[:, i]; s[:, i] = np.where(rng.random(chains) < expit(2 * f), 1.0, -1.0)
        if it >= burn and it % 5 == 0:
            msum += s.sum(0); Csum += s.T @ s; n += chains
    return msum / n, Csum / n

def pl_fit(S, lam=1e-3):
    T, N = S.shape; Y = (S + 1) / 2; W = np.zeros((N, N)); H = np.zeros(N)
    for i in range(N):
        X = np.delete(S, i, 1); y = Y[:, i]
        def f(w):
            z = 2 * (w[0] + X @ w[1:]); r = expit(z) - y
            return (np.logaddexp(0, z) - y * z).mean() + lam * (w[1:] ** 2).sum(), np.concatenate([[2 * r.mean()], 2 * X.T @ r / T + 2 * lam * w[1:]])
        w = minimize(f, np.zeros(N), jac=True, method="L-BFGS-B").x
        H[i] = w[0]; W[i, np.arange(N) != i] = w[1:]
    return H, (W + W.T) / 2

def run(d):
    s = d.split("/")[-4][:6]; st = int(d.rstrip("/").split("_")[-1])
    rec = dict(session=s, stim=st)
    try:
        PRS = pd.read_csv(f"{d}/per_reach_state.csv")
        h = pd.read_csv(f"{d}/h_parameters.csv").h_values.values; N = len(h)
        jv = pd.read_csv(f"{d}/J_parameters.csv").values.ravel(); J = np.zeros((N, N)); iu = np.triu_indices(N, 1)
        cols = pd.read_csv(f"{d}/J_parameters.csv").columns
        ij = [tuple(int(x) - 1 for x in c.split("_")[1:]) for c in cols]; 
        for (a, b), v in zip(ij, jv): J[a, b] = J[b, a] = v
        rec.update(N=N, J_order_is_triu=all(ij[k] == (iu[0][k], iu[1][k]) for k in range(len(ij))))
        spk = PRS[[f"spike_n{i}" for i in range(N)]].values.astype(np.uint8); R = PRS.reach_idx.nunique()
        # (a) against bin10_spike file
        blk = getattr(sio.loadmat(b10[s], squeeze_me=True, struct_as_record=False)["kinAggrogate"], f"stim_{st}")
        ks = sorted([int(n.split("_")[1]) for n in blk._fieldnames if n.startswith("n_")])
        Xb = np.concatenate([(np.asarray(getattr(blk, f"n_{k}"), float)[100:800, 1:] > 0).astype(np.uint8) for k in ks])
        xb = np.concatenate([np.asarray(getattr(blk, f"l_{k}"), float)[100:800, -3] for k in ks])
        rec.update(R=R, R_b10=len(ks), spikes_match=bool(Xb.shape == spk.shape and (Xb == spk).all()),
                   x_maxdiff=float(np.abs(xb - PRS.x.values).max()) if len(xb) == len(PRS) else None,
                   fr_maxdiff=float(np.abs(spk.mean(1) - PRS.firing_rate.values).max()))
        # (b) energy terms
        S = 2.0 * spk - 1; hs = S @ h; jss = 0.5 * np.einsum("ti,ij,tj->t", S, J, S); E = -(hs + jss)
        rec.update(h_maxdiff=float(np.abs(hs - PRS.h.values).max()), j_maxdiff=float(np.abs(jss - PRS.j.values).max()),
                   E_maxdiff=float(np.abs(E - PRS.energy.values).max()))
        # (c) moments
        m_d = S.mean(0); C_d = S.T @ S / len(S)
        m_m, C_m = moments_exact(h, J) if N <= 20 else moments_gibbs(h, J)
        rec.update(moment_method="exact" if N <= 20 else "gibbs", max_dm=float(np.abs(m_m - m_d).max()), rms_dm=float(np.sqrt(((m_m - m_d) ** 2).mean())),
                   max_dC=float(np.abs(C_m - C_d)[iu].max()), rms_dC=float(np.sqrt(((C_m - C_d)[iu] ** 2).mean())),
                   rms_cov_data=float(np.sqrt(((C_d - np.outer(m_d, m_d))[iu] ** 2).mean())),
                   frac_cov_explained=float(1 - (((C_m - C_d)[iu]) ** 2).sum() / (((C_d - np.outer(m_d, m_d))[iu]) ** 2).sum()))
        # independent-model reference: how far the J=0 model is from data correlations
        hi = np.arctanh(np.clip(m_d, -0.999, 0.999)); rec["rms_dC_indep"] = float(np.sqrt(((np.outer(m_d, m_d) - C_d)[iu] ** 2).mean()))
        # (d) PL refit
        Hp, Jp = pl_fit(S); mp, Cp = moments_exact(Hp, Jp) if N <= 20 else moments_gibbs(Hp, Jp, seed=2)
        rec.update(r_J_stored_vs_PL=float(np.corrcoef(J[iu], Jp[iu])[0, 1]), r_h_stored_vs_PL=float(np.corrcoef(h, Hp)[0, 1]),
                   slope_J=float(np.polyfit(Jp[iu], J[iu], 1)[0]), rms_dC_PL=float(np.sqrt(((Cp - C_d)[iu] ** 2).mean())), max_dm_PL=float(np.abs(mp - m_d).max()))
    except Exception as e:
        rec["error"] = repr(e)[:300]
    print(s, st, rec.get("N"), rec.get("error", "ok"), flush=True)
    return rec

dirs = sorted(glob.glob(f"{ROOT}/*_results/*_rep1/full_reach/stim_[012]"))
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    res = p.map(run, dirs)
pd.DataFrame(res).to_csv("verify_fit.csv", index=False); print("done", len(res))
