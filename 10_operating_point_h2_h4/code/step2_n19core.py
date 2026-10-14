"""Parallel 'core' parts for the N=19 conditions (the split job's slow parts). Usage: python step2_n19core.py --npz F --out DIR --key K --ncpu N
Writes DIR/part_<key>_-1_core.pkl (same format/seeds as step2_split.py's core part). Merge with merge_n19.py."""
import argparse, os, pickle, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_lib as L
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); ap.add_argument("--key", required=True); ap.add_argument("--ncpu", type=int, default=6)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); fn = f"{a.out}/part_{a.key}_-1_core.pkl"
if os.path.exists(fn): print("exists", fn); sys.exit(0)
rec = L.analyze_core_pool(dict(np.load(a.npz)), a.key, None, a.ncpu); pickle.dump(rec, open(fn + ".tmp", "wb")); os.replace(fn + ".tmp", fn); print("done", a.key, round(rec["secs"]), flush=True)
