"""fold_k re-check (amendment 7). Usage: python step2_foldk.py --npz F --out DIR --ncpu N [--keys ...] [--folds fold_k|mod5]
One task = (condition or 16-neuron subset, fold f); saved as DIR/foldk_<key>_<sub>_<f>.pkl (resumable). Then
python step2_foldk.py --merge --out DIR --main MAIN_STEP2_DIR --merged OUT_DIR   writes patched task_*.pkl copies for step3."""
import argparse, os, pickle, sys, glob, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz"); ap.add_argument("--out", required=True); ap.add_argument("--ncpu", type=int, default=1)
ap.add_argument("--keys", default="all"); ap.add_argument("--nsub", type=int, default=10); ap.add_argument("--folds", default="fold_k")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--main"); ap.add_argument("--merged")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if a.merge:
    os.makedirs(a.merged, exist_ok=True)
    for f in sorted(glob.glob(f"{a.main}/task_*.pkl")):
        task = pickle.load(open(f, "rb")); key, si = os.path.basename(f)[5:-4].rsplit("_", 1) if False else (None, None)
        base = os.path.basename(f)[5:-4]; key, si = base.rsplit("_", 1)
        parts = [pickle.load(open(g, "rb")) for g in sorted(glob.glob(f"{a.out}/foldk_{key}_{si}_*.pkl"))]
        if len(parts) != 5: print("MISSING fold parts", key, si, len(parts)); continue
        pickle.dump(L.merge_foldk(parts, task), open(f"{a.merged}/{os.path.basename(f)}", "wb"))
    print("merged"); sys.exit(0)
Z = dict(np.load(a.npz)); keys = sorted(k[2:] for k in Z if k.startswith("X_")) if a.keys == "all" else a.keys.split(",")

def run(t):
    key, sub, f = t; fn = f"{a.out}/foldk_{key}_{-1 if sub is None else sub[0]}_{f}.pkl"
    if os.path.exists(fn): return fn
    rec = L.analyze_foldk(Z, key, sub, f, a.folds); tmp = fn + ".tmp"; pickle.dump(rec, open(tmp, "wb")); os.replace(tmp, fn); print("done", key, rec["sub"], f, round(rec["secs"]), flush=True); return fn

if __name__ == "__main__":
    tasks = []
    for k in keys:
        N = Z["X_" + k].shape[2]
        subs = [None] if N <= 20 else [(i, 1000 * i + int(k.replace("_", ""))) for i in range(a.nsub)]
        tasks += [(k, s, f, 2 ** (16 if s else N)) for s in subs for f in range(5)]
    tasks.sort(key=lambda t: -t[3]); print(len(tasks), "tasks", flush=True)
    with mp.get_context("fork").Pool(a.ncpu) as pool: list(pool.imap_unordered(run, [t[:3] for t in tasks]))
    print("all done")
