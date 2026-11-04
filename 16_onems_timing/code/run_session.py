"""E16 real data, one session: 3 conditions x 3 bases (U0, U25, STD). Rows per condition, basis and window (reach, rest shifts).
    python run_session.py SPK_DIR LANDMARKS OUT_DIR SESSION NCPU"""
import sys, os
import numpy as np, pandas as pd
import e16_lib as L

if __name__ == "__main__":
    SPK, LMK, OUT, SESS, NCPU = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
    z = dict(np.load(os.path.join(SPK, f"{SESS}.npz"))); LM = pd.read_csv(LMK); rows, tc = [], {}
    for st in (0, 1, 2):
        m = (z["stim"] == st) & (z["exclude"] == 0); X = (z["c1"][m] > 0).astype(np.uint8); assert X.shape[1] == L.T
        r = LM[(LM.session == int(SESS)) & (LM.stim == st)].iloc[0]; pv, pa = int(r.zero_cross) * 10, int(r.peak_acc) * 10
        for b in L.BASES:
            B = L.basis(b, pv); fit = L.fit_M1(X, B, NCPU); ft = L.field_term(X, fit["h"]); tc[f"{SESS}_{st}__{b}"] = ft.astype(np.float32)
            for where, sh in [("reach", 0)] + [(f"rest{s}", s) for s in L.REST_SHIFTS]:
                rows.append(dict(session=int(SESS), stim=st, key=f"{SESS}_{st}", basis=b, n_basis=B.shape[1], window=where, N=X.shape[2], R=X.shape[0],
                                 pv_ms=pv, pa_ms=pa, converged=fit["converged"], search=str(L.SEARCH), hit_ms=L.HIT_MS, smooth_ms=L.SMOOTH_MS,
                                 **L.search(ft, pv, pa, sh)))
            print(SESS, st, b, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"e16_{SESS}.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, f"e16_ft_{SESS}.npz"), **tc)
