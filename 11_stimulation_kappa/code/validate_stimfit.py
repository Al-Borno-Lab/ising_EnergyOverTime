"""SYNTHETIC method validation of stimfit_model (criteria: ../VALIDATION_CRITERIA_2026-10-12.md).
usage: python validate_stimfit.py <n_procs> <out_dir> [N=12] [Rlist=60:8,18:16]   (R:replicates)
Design as pre-registered (amendment 3): fit bins T_LO:T_HI, control window = per-reach pseudo-onset drawn from the replicate's own stim_1 onsets.
Tasks (scenario x replicate) run in parallel worker processes, one BLAS thread each."""
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stimfit_model as sm
from tvfield_model_basis import bspline_basis

T = 700; W = 6; T_LO, T_HI = 300, 560; SCEN = ["A", "B", "C", "D", "E"]


def truth(N, rng):
    J = rng.normal(0.03, 0.12, (N, N)); J = np.triu(J, 1); J = J + J.T
    h0 = rng.normal(0.3, 0.15, N); c = rng.uniform(0.2, 0.8, N)
    t = np.arange(T)
    H = h0[None] - c[None] * np.exp(-((t[:, None] - 400) / 40.0) ** 2)
    return J, H


def sample_bins(H_rt, J, kap_rt, rng, S, q):
    """H_rt (n,N), kap_rt (n,) -> n states by exact sampling. Groups bins to share work."""
    L = S @ H_rt.T + q[:, None] * (1 + kap_rt)[None]
    L -= L.max(0); P = np.exp(L); P /= P.sum(0)
    cdf = np.cumsum(P, 0); u = rng.random(len(kap_rt))
    idx = np.array([np.searchsorted(cdf[:, j], u[j] * cdf[-1, j]) for j in range(len(u))])
    return (S[np.minimum(idx, len(S) - 1)] > 0).astype(np.uint8)


def simulate(scen, N, seed, R=60):
    rng = np.random.default_rng(1000 * (SCEN.index(scen) + 1) + seed + (0 if R == 60 else 500000))
    J, H = truth(N, rng); S = sm.states(N); q = 0.5 * np.einsum("si,ij,sj->s", S, J, S)
    onsets = rng.integers(400, 402, R)
    Xs = []; pulse = np.zeros((R, T), bool)
    for r in range(R): pulse[r, onsets[r]:onsets[r] + W] = True
    ctrl = np.zeros((R, T), bool)
    pseudo = np.random.default_rng(seed + 90000).choice(onsets, R, replace=True)                       # pseudo-onsets from this replicate's stim_1 onsets
    for r in range(R): ctrl[r, pseudo[r]:pseudo[r] + W] = True
    kap_true = 0.5 if scen in ("A", "E") else 0.0
    for b in range(3):
        Hrt = np.broadcast_to(H, (R, T, N)).copy(); kap = np.zeros((R, T))
        if b == 1:
            if scen in ("B", "E"):
                for r in range(R):
                    k = np.arange(T) - onsets[r]; ramp = np.where(k < 0, 0.0, np.where(k < 30, 0.5 * (1 - np.cos(np.pi * k / 30.0)) , 1.0))
                    Hrt[r] += 0.5 * ramp[:, None]
            if scen == "C": Hrt[pulse] += 0.8
            if scen in ("A", "E"): kap[pulse] = kap_true
        X = np.empty((R, T, N), np.uint8)
        # sample all (r,t) bins in chunks sharing the state table
        flat_H = Hrt.reshape(-1, N); flat_k = kap.reshape(-1)
        out = np.empty((R * T, N), np.uint8)
        for a in range(0, R * T, 2000):
            out[a:a + 2000] = sample_bins(flat_H[a:a + 2000], J, flat_k[a:a + 2000], rng, S, q)
        Xs.append(out.reshape(R, T, N))
    groups = [dict(block=1, mask=pulse, name="pulse"), dict(block=0, mask=ctrl, name="control")]
    return Xs, groups, J, kap_true


def task(args):
    scen, seed, N, R = args
    t0 = time.time(); Xs, groups, J, kt = simulate(scen, N, seed, R)
    B = bspline_basis(400)[T_LO:T_HI]
    Xs = [X[:, T_LO:T_HI] for X in Xs]; groups = [dict(g, mask=g["mask"][:, T_LO:T_HI]) for g in groups]
    res = sm.fit_session(Xs, groups, B, maxiter=800)
    iu = np.triu_indices(N, 1)
    return dict(scen=scen, seed=seed, N=N, R=R, kappa_pulse=float(res["kappa"][0]), kappa_control=float(res["kappa"][1]),
                dkappa=float(res["kappa"][0] - res["kappa"][1]), true=kt, converged=res["converged"], nit=res["nit"],
                Jcorr=float(np.corrcoef(res["J"][iu], J[iu])[0, 1]), secs=time.time() - t0)


if __name__ == "__main__":
    nproc = int(sys.argv[1]); out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    rl = sys.argv[4] if len(sys.argv) > 4 else "60:8,18:16"
    os.makedirs(out, exist_ok=True)
    jobs = [(s, k, N, int(a.split(":")[0])) for a in rl.split(",") for s in SCEN for k in range(int(a.split(":")[1]))]
    with Pool(nproc) as pool:
        rows = []
        for r in pool.imap_unordered(task, jobs):
            rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(out, "validation_rows.json"), "w"), indent=1)
