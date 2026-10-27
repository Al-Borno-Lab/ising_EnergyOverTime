"""EXPLORATORY (regeneration of analyses whose local outputs were lost). One session per call; all fits here.
    python regen_session.py CODE_DIR E12_CODE_DIR spikes_v2.npz phase_landmarks.csv OUTDIR SESSION
Per block (stim 0/1/2), exclusions applied:
  (1) absolute energy terms per neuron: M1 (time-varying fields + J) field/coupling, Ms (stationary) field/coupling,
      population firing rate; rolled so peak velocity = index 400.                                   -> key 'abs'
  (2) E12-style signals (z-scored to baseline): FT (M1 field term), FT0 (M0 field term), POLb (polarization index),
      FR (mean rate); window -600..+1000 ms around the condition's peak acceleration.                 -> key 'ovl'
  (3) field-dip shuffle controls (E12 search window): real, trial shuffle, per-neuron time shift (+-600 ms),
      label shuffle (another session's landmarks); 5 reps each.                                        -> csv rows
  (4) burster/pauser: times of FT trough, FR min, FR max relative to peak velocity.                    -> csv rows"""
import sys, os
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import tvfield_model as tv
import e12_lib as L
Z = dict(np.load(sys.argv[3])); LM = pd.read_csv(sys.argv[4]); OUT = sys.argv[5]; SESS = sys.argv[6]
os.makedirs(OUT, exist_ok=True)
PRE, POST, NREP = 60, 100, 5
ALLKEYS = sorted(k[2:] for k in Z if k.startswith("X_"))

def lm(key):
    s, st = key.split("_"); r = LM[(LM.session == int(s)) & (LM.stim == int(st))].iloc[0]
    return int(r.peak_acc), int(r.zero_cross)

def terms(X, h, J):
    S = 2.0 * X - 1; N = X.shape[2]
    return -(S * h[None]).sum(2).mean(0) / N, -0.5 * np.einsum("rti,ij,rtj->rt", S, J, S).mean(0) / N

def ft_trough(X, pa, zc, h=None):
    if h is None: h, _ = L.fit_fields(X, zc)
    b, _ = L.extremum(L.zscore(L.field_term(X, h), zc), "min", pa + L.WIN[0], pa + L.WIN[1]); return b, h

def one(key):
    s, st = key.split("_"); X = Z["X_" + key][Z["exclude_" + key] == 0].astype(np.uint8); R, T, N = X.shape
    pa, zc = lm(key); sh = lambda a: np.roll(a, 400 - zc)
    # (1) absolute energy
    m1 = tv.fit(X, tv.bspline_basis(zc), "M1"); ms = tv.fit(X, np.ones((T, 1)), "Ms")
    f1, c1 = terms(X, m1["h"], m1["J"]); fs, cs = terms(X, ms["h"], ms["J"])
    absd = np.stack([sh(f1), sh(c1), sh(fs), sh(cs), sh(X.mean((0, 2)))])
    # (2) E12 signals
    h1, h0 = L.fit_fields(X, zc); raw = L.raw_signals(X, zc, h1, h0)
    ovl = np.stack([L.zscore(raw[k], zc)[pa - PRE:pa + POST + 1] for k in ("FT", "FT0", "POLb", "FR")])
    # (3) shuffles of the field-dip timing
    rng = np.random.default_rng([int(s), int(st)]); rows = []
    b, _ = ft_trough(X, pa, zc, h1)
    rows.append(dict(key=key, kind="real", rep=0, dist=abs(b - pa) * 10, hit=int(zc - 20 <= b <= zc)))
    others = [lm(k) for k in ALLKEYS if k.split("_")[0] != s]
    for r in range(NREP):
        Xs = np.stack([X[rng.permutation(R), :, i] for i in range(N)], 2); b, _ = ft_trough(Xs, pa, zc)
        rows.append(dict(key=key, kind="trial_shuffle", rep=r, dist=abs(b - pa) * 10, hit=int(zc - 20 <= b <= zc)))
        lag = rng.integers(-60, 61, size=N); Xt = np.stack([np.roll(X[:, :, i], lag[i], axis=1) for i in range(N)], 2)
        b, _ = ft_trough(Xt, pa, zc)
        rows.append(dict(key=key, kind="neuron_time_shift", rep=r, dist=abs(b - pa) * 10, hit=int(zc - 20 <= b <= zc)))
        opa, ozc = others[rng.integers(len(others))]; b, _ = ft_trough(X, opa, ozc, h1)
        rows.append(dict(key=key, kind="label_shuffle", rep=r, dist=abs(b - opa) * 10, hit=int(ozc - 20 <= b <= ozc)))
    # (4) burster / pauser
    fr = L.zscore(L.sm(X.mean((0, 2))), zc); ft = L.zscore(L.field_term(X, h1), zc)
    lo, hi = pa + L.WIN[0], pa + L.WIN[1]
    tft, _ = L.extremum(ft, "min", lo, hi); tmin, _ = L.extremum(fr, "min", lo, hi); tmax, _ = L.extremum(fr, "max", lo, hi)
    reach = fr[zc - 20:zc + 21].mean()
    mm = dict(key=key, session=s, ft_t=(tft - zc) * 10, frmin_t=(tmin - zc) * 10, frmax_t=(tmax - zc) * 10,
              type="burster" if reach > 0 else "pauser", d_min=abs(tmin - tft) * 10, d_max=abs(tmax - tft) * 10)
    return key, absd, ovl, rows, mm

if __name__ == "__main__":
    keys = [k for k in ALLKEYS if k.split("_")[0] == SESS]
    with Pool(min(len(keys), int(os.environ.get("SLURM_CPUS_PER_TASK", 3)))) as p: res = p.map(one, keys)
    np.savez_compressed(f"{OUT}/regen_{SESS}.npz", keys=np.array([r[0] for r in res]),
                        abs=np.stack([r[1] for r in res]), ovl=np.stack([r[2] for r in res]))
    pd.DataFrame([x for r in res for x in r[3]]).to_csv(f"{OUT}/fielddip_shuffles_{SESS}.csv", index=False)
    pd.DataFrame([r[4] for r in res]).to_csv(f"{OUT}/fr_minmax_{SESS}.csv", index=False)
    print("done", SESS)
