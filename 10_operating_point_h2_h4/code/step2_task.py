"""E10 step 2, ONE task per SLURM array element, parallel inside the task (identical results to step2_real_h2_h4.py).
Usage: python step2_task.py --npz F --out DIR --ncpu N --index I [--list-tasks]   (I = row of the remaining-task list written by --list-tasks)
Task list = all tasks (same definition as step2_real_h2_h4.py) minus those whose task pkl exists in --out or --skip-dirs, minus --exclude-keys, heaviest first."""
import argparse, os, pickle, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ncpu", type=int, default=16)
ap.add_argument("--index", type=int); ap.add_argument("--list-tasks", action="store_true"); ap.add_argument("--skip-dirs", default=""); ap.add_argument("--exclude-keys", default="")
ap.add_argument("--nshuf", type=int, default=20); ap.add_argument("--nsub", type=int, default=10); ap.add_argument("--tasklist", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
Z = dict(np.load(a.npz)); skip = [d for d in a.skip_dirs.split(",") if d] + [a.out]; excl = set(k for k in a.exclude_keys.split(",") if k)
def all_tasks():
    t = []
    for k in sorted(x[2:] for x in Z if x.startswith("X_")):
        N = Z["X_" + k].shape[2]
        if k in excl: continue
        if N <= 20: t.append((k, -1, N))
        else: t += [(k, i, 16) for i in range(a.nsub)]
    return sorted(t, key=lambda x: (-x[2], x[0], x[1]))
if a.list_tasks:
    rem = [t for t in all_tasks() if not any(os.path.exists(f"{d}/task_{t[0]}_{t[1]}.pkl") for d in skip)]
    with open(a.tasklist, "w") as f:
        for t in rem: f.write(f"{t[0]} {t[1]} {t[2]}\n")
    print(len(rem), "remaining tasks written to", a.tasklist); sys.exit(0)
key, si, _ = open(a.tasklist).read().split("\n")[a.index].split(); si = int(si)
fn = f"{a.out}/task_{key}_{si}.pkl"
if any(os.path.exists(f"{d}/task_{key}_{si}.pkl") for d in skip): print("exists, skipping", key, si); sys.exit(0)
sub = None if si < 0 else (si, 1000 * si + int(key.replace("_", "")))
rec = L.analyze_pool(Z, key, sub, a.ncpu, a.nshuf); tmp = fn + ".tmp"; pickle.dump(rec, open(tmp, "wb")); os.replace(tmp, fn); print("done", key, si, round(rec["secs"]), flush=True)
