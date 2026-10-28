"""E15 synthetic validation (method check only; never pooled with real data). One real condition per task:
ground truth from fits to all of its reaches, real R / folds / zero_cross / N. Scenarios:
  S0  J = 0 (M0 truth)          S1  fixed real J (M1 truth)          S2  J x 2 inside window W only
    python synth_e15.py SPIKES_NPZ LANDMARKS_CSV OUTDIR TASK"""
import sys, os
import numpy as np, pandas as pd
import e15_lib as L
tv = L.tv
CONDS = ["210421_0", "210511_1", "210515_2", "220516_0", "220519_1", "210614_0"]   # last one N = 25 (Gibbs path)
Z = dict(np.load(sys.argv[1])); LM = pd.read_csv(sys.argv[2]); OUT = sys.argv[3]; key = CONDS[int(sys.argv[4])]
os.makedirs(OUT, exist_ok=True)
s, st = map(int, key.split("_")); m = Z["exclude_" + key] == 0
Xr = Z["X_" + key][m].astype(np.uint8); fold = Z["fold_" + key][m]; R, T, N = Xr.shape
zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0]); B = tv.bspline_basis(zc)
m0 = tv.fit(Xr, B, "M0"); m1 = tv.fit(Xr, B, "M1")
jW = np.ones(T); jW[L.win_bins(zc, "W")] = 2.0
rows = []
for scen, (h, J, js) in {"S0": (m0["h"], 0 * m1["J"], None), "S1": (m1["h"], m1["J"], None), "S2": (m1["h"], m1["J"], jW)}.items():
    rng = np.random.default_rng([15, s, st, int(scen[1])])
    X = L.sample(h, J, R, rng, jscale=js)
    res = L.analyse_condition(X, fold, zc, rng)
    rows += [dict(scenario=scen, key=key, session=s, stim=st, **r) for r in L.window_stats(res)]
    print(scen, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/synth_{key}.csv", index=False)
