"""Merge the split N=19 job's task files into the main step-2 results dir.
Usage: python merge_step2.py MAIN_DIR SPLIT_DIR. Copies tasks missing from MAIN_DIR; where both exist, checks they agree (rtol 1e-6) and prints MISMATCH otherwise."""
import sys, glob, os, pickle, shutil, numpy as np
main, split = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"{split}/task_*.pkl")):
    g = f"{main}/{os.path.basename(f)}"
    if not os.path.exists(g): shutil.copy(f, g); print("copied", os.path.basename(f)); continue
    a, b = pickle.load(open(f, "rb")), pickle.load(open(g, "rb"))
    bad = [k for k in a if k != "secs" and not isinstance(a[k], str) and k in b and not np.isclose(float(a[k]), float(b[k]), rtol=1e-6, atol=1e-9, equal_nan=True)]
    print("both finished", os.path.basename(f), "MISMATCH " + str(bad) if bad else "identical")
