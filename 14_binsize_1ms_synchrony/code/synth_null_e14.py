"""E14 amendment A2: extended null test T0x (fixed J, no kappa, no gain). One stim-0 condition per task, 3 replicates.
    python synth_null_e14.py SPK_DIR OUT_DIR TASK NCPU"""
import sys, os
import numpy as np, pandas as pd
import e14_lib as L
CONDS = ["210421", "210422", "210511", "210512", "210514", "210515", "210606", "210608", "210614", "210619", "210620",
         "220515", "220516", "220517", "220518", "220519", "220520"]          # stim-0 conditions with N >= 8

if __name__ == "__main__":
    SPK, OUT, task, NCPU = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(OUT, exist_ok=True)
    sess = CONDS[task]; z = dict(np.load(os.path.join(SPK, f"{sess}.npz")))
    X1, cm, em = L.load_block(z, 0); R = X1.shape[0]; assert X1.shape[2] >= 8; B1 = L.basis(1)
    truth = L.fit_M1(X1, B1, NCPU); mask = L.window_mask(cm, em, "E", 1, R); rows = []
    for rep in range(3):
        rng = np.random.default_rng([L.SEED, int(sess), rep])
        Xs = L.simulate(truth["h"], truth["J"], R, rng)
        sB, sE, k2 = L.windows_fixed(Xs, cm, em); sup = L.suppressed(sB, sE); cls = L.pair_classes(sup); d = L.pair_D(sB, sE, sup, rng)
        m1 = L.fit_M1(Xs, B1, NCPU); k = L.fit_kappa(Xs, B1, m1["J"], m1["h"], mask, cls)
        pred = L.predicted_D(m1["h"], m1["J"], sB, sE, k2, cm, em, sup, rng)
        rows.append(dict(scenario="T0x", session=int(sess), rep=rep, N=X1.shape[2], R=R, kappa_other=k["kappa"][0], kappa_supp=k["kappa"][1],
                         kappa_converged=k["converged"], n_pairs_supp=k["n_pairs"][1], D_supp_obs=d["D_supp"], D_supp_pred=pred["D_supp"],
                         N3_resid=d["D_supp"] - pred["D_supp"]))
        print("rep", rep, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"null_{sess}.csv"), index=False)
