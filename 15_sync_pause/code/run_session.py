"""E15 real data, one session (its 3 stim conditions in parallel).
    python run_session.py SPIKES_NPZ LANDMARKS_CSV OUTDIR SESSION"""
import sys, os
from multiprocessing import Pool
import numpy as np, pandas as pd
import e15_lib as L
Z = dict(np.load(sys.argv[1])); LM = pd.read_csv(sys.argv[2]); OUT = sys.argv[3]; SESS = sys.argv[4]
os.makedirs(OUT, exist_ok=True)

def one(key):
    s, st = map(int, key.split("_")); m = Z["exclude_" + key] == 0
    X = Z["X_" + key][m].astype(np.uint8); fold = Z["fold_" + key][m]
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0])
    res = L.analyse_condition(X, fold, zc, np.random.default_rng([15, s, st]))
    rows = [dict(key=key, session=s, stim=st, **r) for r in L.window_stats(res)]
    return key, rows, L.timecourses(res)

if __name__ == "__main__":
    keys = sorted(k[2:] for k in Z if k.startswith("X_") and k[2:].split("_")[0] == SESS)
    assert len(keys) == 3, keys
    with Pool(len(keys)) as p: res = p.map(one, keys)
    pd.DataFrame([r for _, rows, _ in res for r in rows]).to_csv(f"{OUT}/e15_{SESS}.csv", index=False)
    np.savez_compressed(f"{OUT}/e15_tc_{SESS}.npz", **{f"{k}__{n}": v for k, _, tc in res for n, v in tc.items()})
    print("done", SESS)
