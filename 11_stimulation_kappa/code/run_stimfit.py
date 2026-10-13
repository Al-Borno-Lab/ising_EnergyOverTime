"""Real-data run of the stimulation model (pre-registered: PREREGISTRATION_2026-10-12.md).

usage:  python run_stimfit.py <spikes_v2.npz> <laser_onsets.csv> <out_dir> <n_procs> [sessions=all|comma list] [--tasks primary,variants,shuffles,s3,exactcheck]

Task = (session, neuron subset, kind, index). Results are written one JSON per task into <out_dir>/tasks/ (resume-safe).
Kinds:  primary                 real fit, floor onsets, pseudo-onset control            (variant P)
        round / shift1 / fixedctrl / allreach / s3   sensitivity and S3 fits           (variants R, S, F, X, E)
        shuf_<k>, k=0..19       trial-shuffle null refits of the primary design
        exact                   one full exact fit with all N neurons (only for the session named in EXACT_SESSION)
Neuron subsets: N <= NCAP: all neurons, one subset. N > NCAP: NSUB random NCAP-neuron subsets, seed fixed per session.
Subsets, pseudo-onsets and shuffles use seeds recorded in each result row.
"""
import os, sys, json, time, csv
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stimfit_model as sm
from tvfield_model_basis import bspline_basis_crop
import thermo_small

NCAP = 16; NSUB = 10; NSHUF = 10; W = 6; SEED0 = 20261012      # A4: 10 diagnostic shuffles per (session, subset)
PREREG_RANGE = (300, 560)               # PREREGISTRATION section 3: fitted bins 300-559. Run 1 wrongly used (0, 700); the run now refuses to start otherwise.
T_LO, T_HI = PREREG_RANGE
EXACT_SESSION = None                    # set from the command line (--exact <session>)
MAXITER = int(os.environ.get("E11_MAXITER", 800)); TOL = float(os.environ.get("E11_TOL", 1e-7))
LAM = {k: float(os.environ["E11_" + k.upper()]) for k in ("lamB", "lamJ", "lamD", "lamK") if "E11_" + k.upper() in os.environ}
_cache = {}


def load_session(npz, laser_csv, date):
    z = np.load(npz, allow_pickle=True)
    Xs, ex, rn = [], [], []
    for b in range(3):
        Xs.append(z["X_%s_%d" % (date, b)]); ex.append(z["exclude_%s_%d" % (date, b)]); rn.append(z["reachnum_%s_%d" % (date, b)])
    on = {}
    for r in csv.DictReader(open(laser_csv)):
        if r["session"] == date and int(r["block"]) == 1:
            if r["onset_float"] != "":
                on[int(r["reach_row"])] = (float(r["onset_float"]), int(r["onset_bin"]))
    return Xs, ex, on


def session_ids(laser_csv):
    return sorted({r["session"] for r in csv.DictReader(open(laser_csv))})


def onsets_for_variant(of, ob, variant):
    """Integer onset bin per stim_1 reach. primary = bin containing the laser time (floor with float tolerance, from merge_laser_onsets);
    round = nearest bin; shift1 = primary + 1. Control pseudo-onsets are drawn from the SAME array (amendment A3, minor issue 5)."""
    if variant == "round": return np.rint(of).astype(int)
    if variant == "shift1": return ob + 1
    return ob.copy()


def build_groups(R0, R1, R2, on, variant, rng_ctrl):
    """Window groups from stim_1 onsets `on` (length R1). Control (stim_0) windows: per-reach pseudo-onsets drawn from `on`."""
    pulse = np.zeros((R1, 700), bool)
    for r in range(R1): pulse[r, on[r]:min(on[r] + W, 700)] = True
    ctrl = np.zeros((R0, 700), bool)
    if variant == "fixedctrl":
        ctrl[:, 401:401 + W] = True
    else:
        pseudo = rng_ctrl.choice(on, R0, replace=True)
        for r in range(R0): ctrl[r, pseudo[r]:min(pseudo[r] + W, 700)] = True
    groups = [dict(block=1, mask=pulse, name="pulse"), dict(block=0, mask=ctrl, name="control")]
    if variant == "s3":
        half = R1 // 2
        early = pulse.copy(); early[half:] = False; late = pulse.copy(); late[:half] = False
        wash = np.zeros((R2, 700), bool); pseudo2 = rng_ctrl.choice(on, R2, replace=True)
        for r in range(R2): wash[r, pseudo2[r]:min(pseudo2[r] + W, 700)] = True
        groups = [dict(block=1, mask=early, name="pulse_early"), dict(block=1, mask=late, name="pulse_late"),
                  dict(block=0, mask=ctrl, name="control"), dict(block=2, mask=wash, name="washout")]
    return groups


def make_design(Xs, ex, onsets, variant, rng_ctrl, use_excluded=False):
    """Returns (Xs_kept, groups, on1). Reach rows kept: exclude==0 unless use_excluded; stim_1 reaches without a laser time are dropped."""
    keep = [np.ones(len(e), bool) if use_excluded else (e == 0) for e in ex]
    rows1 = np.array([r for r in np.nonzero(keep[1])[0] if r in onsets])
    keep[1] = np.zeros(len(ex[1]), bool); keep[1][rows1] = True
    Xk = [X[k] for X, k in zip(Xs, keep)]
    of = np.array([onsets[r][0] for r in rows1]); ob = np.array([onsets[r][1] for r in rows1])
    on = onsets_for_variant(of, ob, variant)
    groups = build_groups(Xk[0].shape[0], len(on), Xk[2].shape[0], on, variant, rng_ctrl)
    return Xk, groups, on


def crop(Xs, groups):
    Xs = [X[:, T_LO:T_HI] for X in Xs]
    groups = [dict(g, mask=g["mask"][:, T_LO:T_HI]) for g in groups]
    return Xs, groups, bspline_basis_crop(T_LO, T_HI)


def subsets_for(date, N):
    if N <= NCAP: return [np.arange(N)]
    rng = np.random.default_rng([SEED0, int(date)])
    return [np.sort(rng.choice(N, NCAP, replace=False)) for _ in range(NSUB)]


def shuffle_blocks(Xs, rng, on1):
    """Trial-shuffle null (amendment A3). Each neuron's reach order is permuted independently within each block. In stim_1 a neuron's data
    from reach p placed in row r is shifted in time by (on1[r] - on1[p]) bins (circularly; the fitted range 300-559 is far from the wrap), so
    its onset-locked response stays aligned with row r's pulse window. Run 1 permuted without the shift, which smeared the response across the window."""
    out = []
    for b, X in enumerate(Xs):
        R, T, N = X.shape; Y = np.empty_like(X)
        for i in range(N):
            perm = rng.permutation(R)
            if b == 1:
                for r in range(R): Y[r, :, i] = np.roll(X[perm[r], :, i], int(on1[r] - on1[perm[r]]))
            else:
                Y[:, :, i] = X[perm, :, i]
        out.append(Y)
    return out


def run_task(task):
    npz, laser, outdir, date, sub_i, kind, exact = task
    path = os.path.join(outdir, "tasks", "%s_s%d_%s.json" % (date, sub_i, kind))
    if os.path.exists(path): return path
    t0 = time.time()
    Xs, ex, on = load_session(npz, laser, date)
    N = Xs[0].shape[2]
    sub = np.arange(N) if exact else subsets_for(date, N)[sub_i]
    variant = {"primary": "primary"}.get(kind, kind.split("_")[0] if kind.startswith("shuf") else kind)
    design_variant = "primary" if variant.startswith("shuf") or variant == "exact" else variant
    rng_ctrl = np.random.default_rng([SEED0, int(date), 7])           # pseudo-onset seed, same draw for every fit of a session
    assert (T_LO, T_HI) == PREREG_RANGE, "fitted range must equal the pre-registered range"
    Xk, groups, on1 = make_design(Xs, ex, on, design_variant, rng_ctrl, use_excluded=(kind == "allreach"))
    Xk = [X[:, :, sub] for X in Xk]
    seed_shuf = None
    if kind.startswith("shuf"):
        k = int(kind.split("_")[1]); seed_shuf = [SEED0, int(date), sub_i, 100 + k]
        Xk = shuffle_blocks(Xk, np.random.default_rng(seed_shuf), on1)
    Xk, groups, B = crop(Xk, groups)
    res = sm.fit_session(Xk, groups, B, maxiter=MAXITER, tol=TOL, **LAM)
    reran = False
    if not res["converged"]:                                            # pre-registered: rerun once from the last iterate, 2000 iterations
        res = sm.fit_session(Xk, groups, B, p0=res["p"], maxiter=2000, tol=TOL, **LAM); reran = True
    row = dict(role=("diagnostic" if kind.startswith("shuf") else "analysis"), session=date, subset=sub_i, neurons=[int(i) for i in sub], kind=kind, N=int(N), n_used=len(sub),
               groups=res["groups"], kappa=[float(k) for k in res["kappa"]], converged=res["converged"], nit=res["nit"],
               ll=res["ll"], seed_subset=[SEED0, int(date)], seed_shuffle=seed_shuf, seed_pseudo=[SEED0, int(date), 7],
               t_range=[T_LO, T_HI], onset_convention="floor(onset_float+1e-6), t_k = bin start", shuffle="aligned (A3)", maxiter=MAXITER, tol=TOL, lam=LAM, reran=reran, secs=time.time() - t0)
    if kind == "primary" or kind == "exact":
        row.update(thermo_small.window_thermo(Xk, groups, B, res))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(row, open(path, "w"))
    return path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    npz, laser, outdir, nproc = args[0], args[1], args[2], int(args[3])
    sess = session_ids(laser) if len(args) < 5 or args[4] == "all" else args[4].split(",")
    kinds = "primary,round,shift1,fixedctrl,allreach,s3,shuffles"
    exact = None
    for a in sys.argv[1:]:
        if a.startswith("--tasks="): kinds = a.split("=", 1)[1]
        if a.startswith("--exact="): exact = a.split("=", 1)[1]
    kinds = kinds.split(",")
    only_sub = None; nshuf = NSHUF
    for a in sys.argv[1:]:
        if a.startswith("--subsets="): only_sub = [int(x) for x in a.split("=", 1)[1].split(",")]
        if a.startswith("--nshuf="): nshuf = int(a.split("=", 1)[1])
    tasks = []
    z = np.load(npz, allow_pickle=True)
    for d in sess:
        N = z["X_%s_0" % d].shape[2]
        if "exactcheck" in kinds:
            if d == exact: tasks.append((npz, laser, outdir, d, 0, "exact", True))
            continue
        for si in range(len(subsets_for(d, N))):
            if only_sub is not None and si not in only_sub: continue
            for k in kinds:
                if k == "shuffles":
                    tasks += [(npz, laser, outdir, d, si, "shuf_%d" % j, False) for j in range(nshuf)]
                else:
                    tasks.append((npz, laser, outdir, d, si, k, False))
    cost = {d: 4 ** min(z["X_%s_0" % d].shape[2], NCAP) for d in sess}
    tasks.sort(key=lambda t: -cost[t[3]])                              # longest first
    print("tasks:", len(tasks), flush=True)
    os.makedirs(os.path.join(outdir, "tasks"), exist_ok=True)
    with Pool(nproc) as pool:
        for p in pool.imap_unordered(run_task, tasks):
            print("done", os.path.basename(p), flush=True)


if __name__ == "__main__":
    main()
