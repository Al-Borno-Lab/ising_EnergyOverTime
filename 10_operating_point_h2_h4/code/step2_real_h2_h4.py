"""E10 step 2: per-condition H2-H4 quantities on real data. Usage:
python step2_real_h2_h4.py --npz spikes_v2.npz --out DIR --ncpu 64 [--keys k1,k2|all] [--nshuf 20] [--nsub 10]
One task per condition (N <= 20) or per random 16-neuron subset (N > 20; amendment 1). Each task is saved as DIR/task_<key>_<sub>.pkl; finished tasks are skipped on rerun."""
import argparse, os, pickle, sys, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ncpu", type=int, default=1); ap.add_argument("--keys", default="all"); ap.add_argument("--nshuf", type=int, default=20)
ap.add_argument("--nsub", type=int, default=10); ap.add_argument("--max_n", type=int, default=99)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
Z = dict(np.load(a.npz))
keys = sorted(k[2:] for k in Z if k.startswith("X_")) if a.keys == "all" else a.keys.split(",")


def run(t):
    key, sub = t; fn = f"{a.out}/task_{key}_{-1 if sub is None else sub[0]}.pkl"
    if os.path.exists(fn): return fn
    rec = L.analyze(Z, key, sub, nshuf=a.nshuf); pickle.dump(rec, open(fn, "wb")); print("done", key, rec["sub"], round(rec["secs"]), flush=True); return fn

if __name__ == "__main__":
    tasks = []
    for k in keys:
        N = Z["X_" + k].shape[2]
        if N > a.max_n: continue
        if N <= 20: tasks.append((k, None, N))
        else: tasks += [(k, (i, 1000 * i + int(k.replace("_", ""))), 16) for i in range(a.nsub)]
    tasks.sort(key=lambda t: -(2 ** t[2]))                 # heaviest first
    print(len(tasks), "tasks", flush=True)
    with mp.get_context("fork").Pool(a.ncpu) as pool: list(pool.imap_unordered(run, [(k, s) for k, s, _ in tasks]))
    print("all done")
