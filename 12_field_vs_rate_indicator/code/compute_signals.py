"""E12: per-condition signals, extremum search and decoder, from spikes_v2.npz (exclude_k == 0 unless --keep-all).
Writes <out>/rows.csv (one row per condition x signal x window), <out>/traces.npz, <out>/conditions.csv."""
import argparse, os, time, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import numpy as np, pandas as pd
import e12_lib as L

def work(args):
    key, X, fold, pa, zc, sigma = args
    t0 = time.time()
    rows, ex = L.process_condition(X, fold, pa, zc, sigma=sigma)
    for r in rows: r["cond"] = key
    return key, rows, ex, X.shape, time.time() - t0

def load_conditions(npz, landmarks, keep_all=False, min_n=0):
    z = np.load(npz, allow_pickle=True)
    lm = pd.read_csv(landmarks)
    items = []
    for _, r in lm.iterrows():
        key = f"{int(r.session)}_{int(r.stim)}"
        X = z["X_" + key]; fold = z["fold_" + key]; ex = z["exclude_" + key]
        keep = np.ones(len(ex), bool) if keep_all else (ex == 0)
        X, fold = X[keep].astype(np.uint8), fold[keep]
        if X.shape[2] < min_n: continue
        items.append((key, X, fold, int(r.peak_acc), int(r.zero_cross)))
    return items

def run(items, out, sigma=L.SIGMA, nproc=4):
    os.makedirs(out, exist_ok=True)
    jobs = [(k, X, f, pa, zc, sigma) for k, X, f, pa, zc in items]
    allrows, tr, meta = [], {}, []
    with Pool(nproc) as p:
        for key, rows, ex, shp, dt in p.imap_unordered(work, jobs):
            allrows += rows
            for s, v in ex["traces"].items(): tr[f"{key}__{s}"] = v
            meta.append(dict(cond=key, R=shp[0], N=shp[2], dec_r2=ex.get("dec_r2", np.nan), secs=dt))
            print(f"{key} R={shp[0]} N={shp[2]} {dt:.0f}s", flush=True)
    pd.DataFrame(allrows).to_csv(f"{out}/rows.csv", index=False)
    pd.DataFrame(meta).sort_values("cond").to_csv(f"{out}/conditions.csv", index=False)
    np.savez_compressed(f"{out}/traces.npz", **tr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--landmarks", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--min-n", type=int, default=0); ap.add_argument("--sigma", type=float, default=L.SIGMA)
    ap.add_argument("--nproc", type=int, default=4)
    a = ap.parse_args()
    run(load_conditions(a.npz, a.landmarks, a.keep_all, a.min_n), a.out, a.sigma, a.nproc)
