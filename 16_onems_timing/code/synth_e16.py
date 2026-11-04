"""E16 synthetic validation (method check; never pooled with real data). Truth = U0 fit of a real condition, fields smoothed with a
100-ms Gaussian; J real. Z0 = smooth fields; Z1 = Z0 + planted delta_i(t) = 0.3 sigma_i [g(t-pv) - g(t-pa)], g Gaussian SD 15 ms,
sigma_i = sign of the cell's mean spin in the search window of the Z0 simulation.   python synth_e16.py SPK_DIR LANDMARKS OUT_DIR TASK NCPU"""
import sys, os
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import e16_lib as L
CONDS = ["210421_0", "210511_1", "210515_2", "220516_0", "220519_1", "210614_0"]

if __name__ == "__main__":
    SPK, LMK, OUT, task, NCPU = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
    key = CONDS[task]; s, st = map(int, key.split("_")); z = dict(np.load(os.path.join(SPK, f"{s}.npz"))); LM = pd.read_csv(LMK)
    m = (z["stim"] == st) & (z["exclude"] == 0); X = (z["c1"][m] > 0).astype(np.uint8); R, _, N = X.shape
    r = LM[(LM.session == s) & (LM.stim == st)].iloc[0]; pv, pa = int(r.zero_cross) * 10, int(r.peak_acc) * 10
    rng = np.random.default_rng([L.SEED, s, st])
    fit = L.fit_M1(X, L.basis("U0", pv), NCPU); h0 = gaussian_filter1d(fit["h"], 100, axis=0); J = fit["J"]
    X0 = L.simulate(h0, J, R, rng)
    w = np.arange(pv + L.SEARCH[0], pv + L.SEARCH[1] + 1); sig = np.sign((2.0 * X0[:, w] - 1).mean((0, 1))); sig[sig == 0] = 1
    t = np.arange(L.T); g = lambda c: np.exp(-(t - c) ** 2 / (2 * 15.0 ** 2))
    h1 = h0 + 0.3 * sig[None, :] * (g(pv) - g(pa))[:, None]
    X1 = L.simulate(h1, J, R, rng); rows = []
    for scen, Xs, bases in (("Z0", X0, ("U0", "STD")), ("Z1", X1, ("U0",))):
        for b in bases:
            ft = L.field_term(Xs, L.fit_M1(Xs, L.basis(b, pv), NCPU)["h"])
            for where, sh in [("reach", 0)] + [(f"rest{x}", x) for x in L.REST_SHIFTS]:
                rows.append(dict(scenario=scen, key=key, basis=b, window=where, N=N, R=R, **L.search(ft, pv, pa, sh)))
        print(scen, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"synth_{key}.csv"), index=False)
