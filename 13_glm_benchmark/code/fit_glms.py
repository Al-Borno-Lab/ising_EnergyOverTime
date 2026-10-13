"""E13: per-condition CV fits of all GLMs and the trial-shuffle null from spikes_v2.npz (exclude_k == 0 unless --keep-all).
Writes <out>/ll_<cond>.npz: ll_<model> (R,T) held-out LL; null_<model> (nshuf,T) mean-over-reaches LL of the model minus M0 on shuffled data."""
import argparse, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd
import tvfield_model as tv
import glm_lib as G


def work(a):
    key, X, fold, zc, nshuf, out, models, lamj = a
    t0 = time.time()
    B = tv.bspline_basis(zc)
    res = G.cv_models(X, fold, B, zc, models, lamj=lamj)
    d = {f"ll_{m}": v.astype(np.float32) for m, v in res.items()}
    if nshuf:
        rng = np.random.default_rng(1000 + int(key.split("_")[1]) * 100 + int(key.split("_")[0]) % 97)
        nul = {m: [] for m in G.MODELS_NULL if m != "M0"}
        for _ in range(nshuf):
            r = G.cv_models(tv.trial_shuffle(X, rng), fold, B, zc, G.MODELS_NULL, lamj=lamj)
            for m in nul: nul[m].append(np.nanmean(r[m] - r["M0"], 0))
        d.update({f"null_{m}": np.array(v, np.float32) for m, v in nul.items()})
    np.savez_compressed(f"{out}/ll_{key}.npz", **d)
    return key, X.shape, time.time() - t0


def ref_work(a):
    """J-constant parametric-bootstrap reference (amendment A1): fit M1 on all kept reaches, simulate nrep datasets with the same
    fields and J (constant), run M0 and SYNC CV on each, return the mean per-bin gain (T,)."""
    key, X, fold, zc, nrep, out = a
    t0 = time.time(); B = tv.bspline_basis(zc)
    P = tv.fit(X, B, "M1"); rng = np.random.default_rng(777 + int(key.split("_")[0]) % 997 + int(key.split("_")[1]))
    gains = []
    for _ in range(nrep):
        Xs = G.sample_m1(P["h"], P["J"], X.shape[0], rng)
        r = G.cv_models(Xs, fold, B, zc, ["M0", "SYNC"])
        gains.append(np.nanmean(r["SYNC"] - r["M0"], 0))
    np.savez_compressed(f"{out}/ref_{key}.npz", refgain=np.mean(gains, 0).astype(np.float32), refgain_all=np.array(gains, np.float32))
    return key, X.shape, time.time() - t0


def run_ref(jobs, nproc):
    jobs = sorted(jobs, key=lambda j: -j[1].shape[2])
    with Pool(nproc) as p:
        for key, shp, dt in p.imap_unordered(ref_work, jobs):
            print(f"ref {key} R={shp[0]} N={shp[2]} {dt:.0f}s", flush=True)


def run(jobs, nproc):
    jobs = sorted(jobs, key=lambda j: -j[1].shape[2])           # slowest first
    with Pool(nproc) as p:
        for key, shp, dt in p.imap_unordered(work, jobs):
            print(f"{key} R={shp[0]} N={shp[2]} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--landmarks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--keep-all", action="store_true"); ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--nshuf", type=int, default=20); ap.add_argument("--lamj", type=float, default=G.LAMJ)
    ap.add_argument("--models", default=",".join(G.MODELS_ALL)); ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--only", default=""); ap.add_argument("--ref", type=int, default=0, help="J-constant reference replicates (amendment A1); 0 = normal run")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    z = np.load(a.npz, allow_pickle=True); lm = pd.read_csv(a.landmarks)
    jobs = []
    for _, r in lm.iterrows():
        key = f"{int(r.session)}_{int(r.stim)}"
        if a.only and key not in a.only.split(","): continue
        X = z["X_" + key]; fold = z["fold_" + key]; ex = z["exclude_" + key]
        keep = np.ones(len(ex), bool) if a.keep_all else ex == 0
        X, fold = X[keep].astype(np.uint8), fold[keep]
        if X.shape[2] < a.min_n: continue
        jobs.append((key, X, fold, int(r.zero_cross), a.nshuf, a.out, a.models.split(","), a.lamj))
    if a.ref:
        run_ref([(k, X, fo, zc, a.ref, o) for k, X, fo, zc, _n, o, _m, _l in jobs], a.nproc)
    else:
        run(jobs, a.nproc)
