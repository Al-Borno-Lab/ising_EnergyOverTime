"""Build task_<key>_-1.pkl for the N=19 conditions from shuffle chunks (split job) and a core part (split job's own core or the parallel one).
Usage: python merge_n19.py SPLIT_DIR CORE_DIR OUT_DIR KEY[,KEY...]. If both cores exist, checks they agree (rtol 1e-6) and prints MISMATCH otherwise."""
import sys, os, pickle, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
split, coredir, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
for k in sys.argv[4].split(","):
    shuf = [pickle.load(open(f"{split}/part_{k}_-1_{c}.pkl", "rb")) for c in range(4)]
    cores = [pickle.load(open(f, "rb")) for f in (f"{split}/part_{k}_-1_core.pkl", f"{coredir}/part_{k}_-1_core.pkl") if os.path.exists(f)]
    if not cores: print("NO CORE for", k); continue
    if len(cores) == 2:
        bad = [x for x in cores[0] if x not in ("secs",) and not isinstance(cores[0][x], str) and not np.isclose(float(cores[0][x]), float(cores[1][x]), rtol=1e-6, atol=1e-9, equal_nan=True)]
        print(k, "two cores:", "MISMATCH " + str(bad) if bad else "identical")
    pickle.dump(L.merge_parts(shuf + [cores[0]]), open(f"{out}/task_{k}_-1.pkl", "wb")); print("merged", k)
