"""E14 synthetic validation (method check only; never pooled with real data). Ground truth = 1-ms M1 fit of a real stim-0 condition,
real R and windows; cell classes for planting from the real data, classes for analysis re-derived from the synthetic data (as in real use).
  T0 kappa = 0;  T1 kappa_s = +0.5 in E;  T2 kappa = 0 + shared per-reach field offset N(0, 0.3) in E.
    python synth_e14.py SPK_DIR OUT_DIR TASK NCPU"""
import sys, os
import numpy as np, pandas as pd
import e14_lib as L
CONDS = ["210512", "210421", "220516", "210614"]      # N = 8, 15, 19, 25

if __name__ == "__main__":
    SPK, OUT, task, NCPU = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
    sess = CONDS[task]; z = dict(np.load(os.path.join(SPK, f"{sess}.npz")))
    X1, cm, em = L.load_block(z, 0); R = X1.shape[0]; B1 = L.basis(1)
    truth = L.fit_M1(X1, B1, NCPU)
    XB, XE, keep = L.windows_fixed(X1, cm, em); cls_true = L.pair_classes(L.suppressed(XB, XE))
    mask = L.window_mask(cm, em, "E", 1, R); rows = []
    for scen, (kap, gain) in {"T0": (0.0, 0.0), "T1": (0.5, 0.0), "T2": (0.0, 0.3)}.items():
        rng = np.random.default_rng([L.SEED, int(sess), int(scen[1])])
        Jx = kap * np.where(cls_true == 1, truth["J"], 0.0)
        Xs = L.simulate(truth["h"], truth["J"], R, rng, mask=mask, Jextra=Jx if kap else None, gain_sd=gain)
        sB, sE, k2 = L.windows_fixed(Xs, cm, em); sup = L.suppressed(sB, sE); cls = L.pair_classes(sup)
        d = L.pair_D(sB, sE, sup, rng)
        m1 = L.fit_M1(Xs, B1, NCPU); k = L.fit_kappa(Xs, B1, m1["J"], m1["h"], mask, cls)
        pred = L.predicted_D(m1["h"], m1["J"], sB, sE, k2, cm, em, sup, rng)
        rows.append(dict(scenario=scen, session=int(sess), N=X1.shape[2], R=R, planted_kappa_s=kap, gain_sd=gain,
                         class_agreement=float((cls == cls_true).mean()), kappa_other=k["kappa"][0], kappa_supp=k["kappa"][1],
                         kappa_converged=k["converged"], n_pairs_supp=k["n_pairs"][1], D_supp_obs=d["D_supp"], D_supp_pred=pred["D_supp"],
                         D_all_obs=d["D_all"], D_all_pred=pred["D_all"], N3_resid=d["D_supp"] - pred["D_supp"]))
        print(scen, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"synth_{sess}.csv"), index=False)
