"""Tests step3_stats.py on SYNTHETIC task records with known planted truth (mechanics test only; not data).
Usage: python test_step3_synthetic.py SCENARIO OUTDIR   (SCENARIO: supported | kappa_change | refuted_h1 | partial)"""
import sys, os, pickle, numpy as np, pandas as pd, shutil
scen, work = sys.argv[1], sys.argv[2]
E = pd.read_csv("assets/tables/expected_conditions.csv", dtype={"key": str, "session": str})
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1); res = f"{work}/res"; shutil.rmtree(work, ignore_errors=True); os.makedirs(res); os.makedirs(f"{work}/tables"); os.makedirs(f"{work}/figures")
E.to_csv(f"{work}/tables/expected_conditions.csv", index=False)
sess_eff = {s: rng.normal(0, .5) for s in E.session.unique()}
for _, r in E.iterrows():
    for sub in (range(10) if r.N_full > 20 else [-1]):
        if scen == "partial" and r.key in ("210606_0", "220517_1") and sub >= 4: continue
        dv = rng.normal(0.02 if scen != "refuted_h1" else -0.02, 0.02) + 0.0 * sess_eff[r.session]
        dI = (0.8 * dv + rng.normal(0, 0.003)) * (1 if scen != "refuted_h1" else 1)
        chi = rng.normal(0, .1); emp = chi + rng.normal(0, .03)
        kap = rng.normal(0.4 if scen == "kappa_change" else 0, 0.1); gain = rng.normal(0.01 if scen == "kappa_change" else 0, 0.002)
        rec = dict(key=r.key, sub=sub, N_full=r.N_full, N=min(r.N_full, 16 if r.N_full > 20 else r.N_full), R=17, R_all=r.R_all, converged=True, real_dI=dI, real_dchiJ=chi,
                   real_dC=rng.normal(0, .01), real_dv=dv, real_I_base=.02, real_v_base=.2, real_rho=-.5, emp_dcov=emp, emp_dv=dv, null_dI_mean=0.0, null_dI_p95=0.004,
                   null_dI_max=.006, null_rho_mean=-.5, null_dchiJ_mean=0.0, kappa_hat=kap, kappa_converged=not (scen == "partial" and r.key == "210421_0"),
                   kappa_gain_reach=gain, pred_dchiJ_folds=chi + rng.normal(0, .02), secs=1.0)
        pickle.dump(rec, open(f"{res}/task_{r.key}_{sub}.pkl", "wb"))
