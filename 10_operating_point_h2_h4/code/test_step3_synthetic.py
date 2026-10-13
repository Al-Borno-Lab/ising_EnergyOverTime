"""Tests step3_stats.py on SYNTHETIC task records with planted truth (mechanics test only; not data), with assertions.
Usage (from the experiment folder): python code/test_step3_synthetic.py WORKDIR     (exit code 1 if any assertion fails)"""
import sys, os, pickle, shutil, subprocess, numpy as np, pandas as pd
work = sys.argv[1]; E = pd.read_csv("assets/tables/expected_conditions.csv", dtype={"key": str, "session": str})
HERE = os.path.dirname(os.path.abspath(__file__))

def make(scen, seed):
    rng = np.random.default_rng(seed); d = f"{work}/{scen}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(f"{d}/res"); os.makedirs(f"{d}/tables"); os.makedirs(f"{d}/figures")
    E.to_csv(f"{d}/tables/expected_conditions.csv", index=False)
    for _, r in E.iterrows():
        if scen == "missing_session" and r.session == "210423": continue
        for sub in (range(10) if r.N_full > 20 else [-1]):
            if scen == "partial" and r.key in ("210606_0", "220517_1") and sub >= 4: continue
            c22 = r.session.startswith("22")
            dvm = {"refuted_h1": -0.02, "cohort2022only": (0.03 if c22 else 0.0)}.get(scen, 0.02)
            dvs = 0.005 if (scen == "cohort2022only" and c22) else 0.02
            if scen == "c22_5of6_c21_neg":      # reviewer's dodge case: 2022 5/6 positive; 2021 median negative but 6 strongly positive sessions
                i21 = sorted(E.session[E.session.str.startswith("21")].unique()).index(r.session) if not c22 else None
                dvm = (0.03 if r.session != "220520" else -0.02) if c22 else (0.04 if i21 < 6 else -0.01); dvs = 0.003
            if scen == "both_pos_c21_sig": dvm, dvs = (0.03, 0.005) if c22 else (0.02, 0.01)
            if scen == "c21_only": dvm, dvs = (0.0, 0.003) if c22 else (0.02, 0.01)
            dv = rng.normal(dvm, dvs); dI = 0.8 * dv + rng.normal(0, 0.003)
            chi = rng.normal(0, .1); emp = chi + rng.normal(0, .03)
            kap = rng.normal(0.4 if scen == "kappa_change" else 0, 0.1)
            gm = {"kappa_change": 0.01, "pos_gain": 0.003, "neg_gain": -0.002}.get(scen, -0.0003)
            gain = rng.normal(gm, 0.0003)
            p95 = 0.05 if scen == "weak_gate" and r.session[-1] not in "13" else 0.004          # weak_gate: bound far above most sessions' dI
            rec = dict(key=r.key, sub=sub, N_full=r.N_full, N=min(r.N_full, 16 if r.N_full > 20 else r.N_full), R=17, R_all=r.R_all, converged=True, real_dI=dI, real_dchiJ=chi,
                       real_dC=rng.normal(0, .01), real_dv=dv, real_I_base=.02, real_v_base=.2, real_rho=-.5, emp_dcov=emp, emp_dv=dv, null_dI_mean=0.0, null_dI_p95=p95,
                       null_dI_max=.006, null_rho_mean=-.5, null_dchiJ_mean=0.0, kappa_hat=kap, kappa_converged=not (scen == "partial" and r.key == "210421_0"),
                       kappa_gain_reach=gain, pred_dchiJ_folds=chi + rng.normal(0, .02), secs=1.0)
            if scen == "nan" and r.key == "210421_0": rec["real_dI"] = np.nan
            pickle.dump(rec, open(f"{d}/res/task_{r.key}_{sub}.pkl", "wb"))
    p = subprocess.run([sys.executable, f"{HERE}/step3_stats.py", f"{d}/res", d], capture_output=True, text=True)
    if p.returncode: print(p.stderr[-1500:]); raise SystemExit(f"step3 crashed on {scen}")
    t = pd.read_csv(f"{d}/tables/e10_tests.csv"); v = t[(t.set == "primary") & (t.test == "VERDICT")].iloc[0]
    return v, t, open(f"{d}/tables/e10_warnings.txt").read()

CASES = [  # scenario, seed, assertions on (verdict row v, warnings w)
    ("supported", 2, lambda v, w: v.VERDICT_STATUS == "SUPPORTED"),
    ("neg_gain", 2, lambda v, w: v.VERDICT_STATUS == "SUPPORTED" and bool(v.H3b_no_change)),            # negative gain must NOT refute fixed J (R2)
    ("pos_gain", 2, lambda v, w: not bool(v.H3b_no_change) and v.VERDICT_STATUS.startswith("REFUTED")),
    ("kappa_change", 2, lambda v, w: not bool(v.H3b_no_change) and v.VERDICT_STATUS.startswith("REFUTED")),
    ("refuted_h1", 2, lambda v, w: "dI <= 0" in str(v.refuted_because) and v.VERDICT_STATUS.startswith("REFUTED")),
    ("cohort2022only", 3, lambda v, w: bool(v.rule_2022_only_triggered) and v.VERDICT_STATUS.startswith("REFUTED") and "2022 cohort" in str(v.refuted_because)),  # R1
    ("c22_5of6_c21_neg", 2, lambda v, w: bool(v.H2_supported) and bool(v.rule_2022_only_triggered) and v.VERDICT_STATUS.startswith("REFUTED") and v.H2_2021_median <= 0),   # F1 (a)
    ("both_pos_c21_sig", 2, lambda v, w: (not bool(v.rule_2022_only_triggered)) and v.VERDICT_STATUS == "SUPPORTED" and v.H2_2021_p < .05),                      # F1 (b)
    ("c21_only", 2, lambda v, w: (not bool(v.rule_2022_only_triggered)) and v.H2_2022_median is not None),                                                      # F1 (c): no cohort refutation
    ("weak_gate", 2, lambda v, w: not bool(v.H2_supported) and v.VERDICT_STATUS.startswith("REFUTED")),   # significant dI but few sessions above null (R4)
    ("partial", 2, lambda v, w: v.VERDICT_STATUS.startswith("INCOMPLETE") and "210606_0" in w and "non-converged" in w),
    ("nan", 2, lambda v, w: v.VERDICT_STATUS.startswith("INCOMPLETE") and "NaN in" in w),
    ("missing_session", 2, lambda v, w: v.VERDICT_STATUS.startswith("INCOMPLETE") and "210423" in w),
]
fails = 0
for scen, seed, check in CASES:
    v, t, w = make(scen, seed); ok = bool(check(v, w)); fails += not ok
    print(("PASS" if ok else "FAIL"), scen, "| status:", v.VERDICT_STATUS, "| H2", v.H2_supported, "H3a", v.H3a_supported, "H3b_ok", v.H3b_no_change, "H4a", v.H4a_supported, "|", str(v.get("refuted_because", ""))[:90])
sys.exit(1 if fails else 0)
