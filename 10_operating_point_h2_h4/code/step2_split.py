"""CONTINGENCY ONLY (do not run unless step2 hit its time limit). Resubmits unfinished conditions split into parts:
one 'core' part (real fit, kappa, CV) and 4 shuffle chunks; identical seeds/subsets/statistics as step2_real_h2_h4.py.
Usage: python step2_split.py --npz F --out DIR(the step2 dir) --ncpu N [--keys k1,k2|all]
Skips tasks whose task_<key>_<sub>.pkl exists; writes part_*.pkl and merges them into task_*.pkl when all 5 parts exist."""
import argparse, os, pickle, sys, glob, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ncpu", type=int, default=1); ap.add_argument("--keys", default="all"); ap.add_argument("--nshuf", type=int, default=20)
ap.add_argument("--nsub", type=int, default=10); ap.add_argument("--max_n", type=int, default=99)
a = ap.parse_args(); Z = dict(np.load(a.npz))
keys = sorted(k[2:] for k in Z if k.startswith("X_")) if a.keys == "all" else a.keys.split(",")
PARTS = ["core", 0, 1, 2, 3]

def run(t):
    key, sub, part = t; si = -1 if sub is None else sub[0]; fn = f"{a.out}/part_{key}_{si}_{part}.pkl"
    if not os.path.exists(fn): pickle.dump(L.analyze_part(Z, key, sub, part, nshuf=a.nshuf), open(fn, "wb")); print("part", key, si, part, flush=True)
    return t

if __name__ == "__main__":
    tasks = []
    for k in keys:
        N = Z["X_" + k].shape[2]
        if N > a.max_n: continue
        subs = [None] if N <= 20 else [(i, 1000 * i + int(k.replace("_", ""))) for i in range(a.nsub)]
        for s in subs:
            if os.path.exists(f"{a.out}/task_{k}_{-1 if s is None else s[0]}.pkl"): continue
            tasks += [(k, s, p, 2 ** (16 if s else N)) for p in PARTS]
    tasks.sort(key=lambda t: -t[3]); print(len(tasks), "parts", flush=True)
    with mp.get_context("fork").Pool(a.ncpu) as pool: list(pool.imap_unordered(run, [t[:3] for t in tasks]))
    for k in keys:
        for f in glob.glob(f"{a.out}/part_{k}_*_core.pkl"):
            si = f.split("_")[-2]; ps = [pickle.load(open(g, "rb")) for g in glob.glob(f"{a.out}/part_{k}_{si}_*.pkl")]
            if len(ps) == 5 and not os.path.exists(f"{a.out}/task_{k}_{si}.pkl"): pickle.dump(L.merge_parts(ps), open(f"{a.out}/task_{k}_{si}.pkl", "wb"))
    print("all done")
