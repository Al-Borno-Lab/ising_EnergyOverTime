"""E14 real data, one session. N1 (1-ms SI, all blocks; primary stim 0), N2 (epoch kappa at 1/5/10/20 ms, stim 0, windows E and A),
N3 (fixed-J predicted SI change, 1 ms, stim 0).      python run_session.py SPK_DIR OUT_DIR SESSION NCPU"""
import sys, os, json
import numpy as np, pandas as pd
import e14_lib as L

if __name__ == "__main__":
    SPK, OUT, SESS, NCPU = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
    z = dict(np.load(os.path.join(SPK, f"{SESS}.npz"))); rng = np.random.default_rng([L.SEED, int(SESS)]); rows = []
    base = dict(session=int(SESS), windows=str(L.WIN_MS), nshuf=L.NSHUF)
    # ---- N1: all blocks, windows E (primary) and A
    cls0 = None
    for st in (0, 1, 2):
        X1, cm, em = L.load_block(z, st)
        if X1.shape[0] < 2: continue
        for test in ("E", "A"):
            XB, XT, keep = L.windows_fixed(X1, cm, em, test)
            if len(keep) < 2: continue
            sup = L.suppressed(XB, L.windows_fixed(X1, cm, em, "E")[1]) if test == "A" else L.suppressed(XB, XT)
            d = L.pair_D(XB, XT, sup, rng)
            rows.append(dict(base, kind="N1", stim=st, bin_ms=1, window=test, N=X1.shape[2], R=X1.shape[0], R_used=len(keep), n_sup_cells=int(sup.sum()), **d))
            if st == 0 and test == "E": sup0, XB0, XE0, keep0, cm0, em0, X10 = sup, XB, XT, keep, cm, em, X1
    # ---- N2: stim 0, per bin size: M1 then kappa for windows E and A (classes from 1-ms stim-0 E vs B)
    cls = L.pair_classes(sup0); store = {}
    for b in L.BIN_SIZES:
        X = L.rebin(X10, b); B = L.basis(b); m1 = L.fit_M1(X, B, NCPU); store[f"J_{b}"] = m1["J"]
        if b == 1: m1_1 = m1
        for w in ("E", "A"):
            mask = L.window_mask(cm0, em0, w, b, X.shape[0])
            k = L.fit_kappa(X, B, m1["J"], m1["h"], mask, cls)
            rows.append(dict(base, kind="N2", stim=0, bin_ms=b, window=w, N=X.shape[2], R=X.shape[0], window_bins=int(mask.sum()),
                             kappa_other=k["kappa"][0], kappa_supp=k["kappa"][1], n_pairs_other=k["n_pairs"][0], n_pairs_supp=k["n_pairs"][1],
                             kappa_converged=k["converged"], kappa_nit=k["nit"], m1_converged=m1["converged"]))
        print("N2 done", b, flush=True)
    # ---- N3: fixed-J prediction of D at 1 ms (stim 0, window E)
    pred = L.predicted_D(m1_1["h"], m1_1["J"], XB0, XE0, keep0, cm0, em0, sup0, rng)
    obs = next(r for r in rows if r["kind"] == "N1" and r["stim"] == 0 and r["window"] == "E")
    rows.append(dict(base, kind="N3", stim=0, bin_ms=1, window="E", N=X10.shape[2], R_used=len(keep0),
                     D_supp_obs=obs["D_supp"], D_other_obs=obs["D_other"], D_all_obs=obs["D_all"],
                     D_supp_pred=pred["D_supp"], D_other_pred=pred["D_other"], D_all_pred=pred["D_all"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"e14_{SESS}.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, f"e14_J_{SESS}.npz"), sup0=sup0, **store)
    print("done", SESS, flush=True)
