"""SYNTHETIC realistic validation of the real-data pipeline (VALIDATION_CRITERIA amendment 6).
usage: python validate_realistic.py <spikes_v2.npz> <laser_onsets_v2.csv> <out_dir> <n_procs> [reps=8]
Synthetic sessions copy the SHAPE of real sessions (reach counts per block, real stim_1 onset bins, real smoothed PSTHs as block-specific fields)
and add a per-reach common gain (rate confound). Each dataset is fitted with the real pipeline (run_stimfit.make_design/build_groups/crop, range 300-559)
plus NNULL aligned shuffle refits."""
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = "1"
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stimfit_model as sm
import run_stimfit as rs
import validate_stimfit as v1

STRUCT = os.environ.get("E11_STRUCT", "210421,210511,210514").split(","); SCEN = "ABCDE"; NNULL = 10; GAIN_SD = 0.25; T = 700


def structure(npz, laser, date):
    Xs, ex, on = rs.load_session(npz, laser, date)
    Xk, groups, on1 = rs.make_design(Xs, ex, on, "primary", np.random.default_rng(0))
    N = Xk[0].shape[2]; sub = np.sort(np.random.default_rng([7, int(date)]).choice(N, min(N, 12), replace=False))
    Xk = [X[:, :, sub] for X in Xk]
    H = [np.arctanh(np.clip(2 * gaussian_filter1d(X.mean(0).astype(float), 3, axis=0, mode="nearest") - 1, -0.95, 0.95)) for X in Xk]
    return dict(date=date, R=[X.shape[0] for X in Xk], on1=on1, H=H, N=len(sub))


def simulate(st, scen, rep):
    rng = np.random.default_rng([1234, SCEN.index(scen), int(st["date"]), rep]); N = st["N"]
    J = rng.normal(0.03, 0.10, (N, N)); J = np.triu(J, 1); J = J + J.T
    S = sm.states(N); q = 0.5 * np.einsum("si,ij,sj->s", S, J, S); on1 = st["on1"]
    kap_true = 0.5 if scen in "AE" else 0.0
    Xs = []
    for b in range(3):
        R = st["R"][b]
        Hrt = np.broadcast_to(st["H"][b], (R, T, N)).copy() + rng.normal(0, GAIN_SD, R)[:, None, None]
        kap = np.zeros((R, T))
        if b == 1:
            for r in range(R):
                k = np.arange(T) - on1[r]
                if scen in "BE": Hrt[r] += 0.5 * np.where(k < 0, 0.0, np.where(k < 30, 0.5 * (1 - np.cos(np.pi * k / 30.0)), 1.0))[:, None]
                if scen == "C": Hrt[r, on1[r]:on1[r] + 6] += 0.8
                if scen in "AE": kap[r, on1[r]:on1[r] + 6] = kap_true
        fh = Hrt.reshape(-1, N); fk = kap.reshape(-1); out = np.empty((R * T, N), np.uint8)
        for a in range(0, R * T, 2000): out[a:a + 2000] = v1.sample_bins(fh[a:a + 2000], J, fk[a:a + 2000], rng, S, q)
        Xs.append(out.reshape(R, T, N))
    return Xs, J, kap_true


def fit_one(Xs, on1, seed_tag, shuffle_k=None):
    Xin = Xs
    if shuffle_k is not None: Xin = rs.shuffle_blocks(Xs, np.random.default_rng(seed_tag + [100 + shuffle_k]), on1)
    rng_ctrl = np.random.default_rng(seed_tag + [7])
    groups = rs.build_groups(Xs[0].shape[0], Xs[1].shape[0], Xs[2].shape[0], on1, "primary", rng_ctrl)
    Xc, gc, B = rs.crop(Xin, groups)
    return sm.fit_session(Xc, gc, B, maxiter=rs.MAXITER, tol=rs.TOL, **rs.LAM)


def task(args):
    npz, laser, date, scen, rep = args
    t0 = time.time(); st = structure(npz, laser, date); Xs, J, kt = simulate(st, scen, rep)
    tag = [20261013, SCEN.index(scen), int(date), rep]
    real = fit_one(Xs, st["on1"], tag)
    nulls = [fit_one(Xs, st["on1"], tag, k) for k in range(NNULL)]
    nk = np.array([n["kappa"][:2] for n in nulls]); kp, kc = real["kappa"][:2]
    iu = np.triu_indices(st["N"], 1)
    return dict(scen=scen, date=date, rep=rep, N=st["N"], R=st["R"], true=kt, kappa_pulse=float(kp), kappa_control=float(kc), d_raw=float(kp - kc),
                null_mean=[float(x) for x in nk.mean(0)], null_sd=[float(x) for x in nk.std(0, ddof=1)],
                d_corr=float((kp - nk[:, 0].mean()) - (kc - nk[:, 1].mean())), converged=bool(real["converged"] and all(n["converged"] for n in nulls)),
                n_conv=int(real["converged"]) + sum(int(n["converged"]) for n in nulls), n_fits=1 + NNULL,
                n_below_m1=int((np.concatenate([[kp, kc], nk.ravel()]) < -1).sum()), Jcorr=float(np.corrcoef(real["J"][iu], J[iu])[0, 1]), secs=time.time() - t0)


if __name__ == "__main__":
    npz, laser, out, nproc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); reps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    assert (rs.T_LO, rs.T_HI) == rs.PREREG_RANGE
    os.makedirs(out, exist_ok=True)
    jobs = [(npz, laser, d, s, r) for d in STRUCT for s in SCEN for r in range(reps)]
    rows = []
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(task, jobs):
            rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(out, "validation_rows.json"), "w"), indent=1)
