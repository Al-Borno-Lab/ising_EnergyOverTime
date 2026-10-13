"""E10 step 2 sensitivity analyses (amendment 7), real fits only. Usage:
python step2_sens.py --npz F --out DIR --ncpu N --mode allreaches|subset16|late [--keys ...]
 allreaches: all reaches (no exclusion), same subsets as the main run for N > 20.
 subset16:   conditions with N = 17-19, exclusion applied, 10 random 16-neuron subsets (compare with the full-population main-run values).
 late:       exclusion applied, same subsets as the main run, adds the exploratory late window (+50..+150).
Each task saved as DIR/sens_<mode>_<key>_<sub>.pkl (resumable)."""
import argparse, os, pickle, sys, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", required=True)
ap.add_argument("--ncpu", type=int, default=1); ap.add_argument("--keys", default="all"); ap.add_argument("--nsub", type=int, default=10)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); Z = dict(np.load(a.npz))
keys = sorted(k[2:] for k in Z if k.startswith("X_")) if a.keys == "all" else a.keys.split(",")

def run(t):
    key, sub = t; fn = f"{a.out}/sens_{a.mode}_{key}_{-1 if sub is None else sub[0]}.pkl"
    if os.path.exists(fn): return fn
    rec = L.analyze_real(Z, key, sub, exclude=(a.mode != "allreaches")); rec["mode"] = a.mode
    tmp = fn + ".tmp"; pickle.dump(rec, open(tmp, "wb")); os.replace(tmp, fn); print("done", key, rec["sub"], round(rec["secs"]), flush=True); return fn

if __name__ == "__main__":
    tasks = []
    for k in keys:
        N = Z["X_" + k].shape[2]
        if a.mode == "subset16":
            if 17 <= N <= 19: tasks += [(k, (i, 1000 * i + int(k.replace("_", "")) + 777), N) for i in range(a.nsub)]
        elif N <= 20: tasks.append((k, None, N))
        else: tasks += [(k, (i, 1000 * i + int(k.replace("_", ""))), N) for i in range(a.nsub)]
    tasks.sort(key=lambda t: -(2 ** (16 if t[1] else t[2]))); print(len(tasks), "tasks", flush=True)
    with mp.get_context("fork").Pool(a.ncpu) as pool: list(pool.imap_unordered(run, [(k, s) for k, s, _ in tasks]))
    print("all done")
