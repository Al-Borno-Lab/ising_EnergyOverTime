"""Assertion fixtures for analyze_stimfit.py (independent review E11_ANALYSIS_REVIEW.md, amendment A5): planted effects, equivalence, null,
small-but-significant, (a)+(b) overlap, missing sessions, section-3 failures, NaN kappa, run-1 rows, kappa < -1 counting. FAKE data only."""
import os, json, shutil, tempfile
import numpy as np
import analyze_stimfit as A


def make(root, delta=0.0, between=0.1, seed=0, drop=(), nonconverge=None, nan_in=None, run1=False, m1_converged=None, m1_nonconverged=None):
    rng = np.random.default_rng(seed); shutil.rmtree(root, ignore_errors=True); os.makedirs(os.path.join(root, "tasks"))
    for s in A.MOUSE:
        if s in drop: continue
        ds = delta + rng.normal(0, between)
        for sub in range(A.n_subsets(s)):
            kc = rng.normal(0, 0.15); kp = kc + ds + rng.normal(0, 0.05)
            for kind in A.CORE_KINDS + tuple("shuf_%d" % j for j in range(A.NSHUF)):
                r = dict(session=s, subset=sub, kind=kind, N=A.NEURONS[s], n_used=min(16, A.NEURONS[s]), converged=True, reran=False, nit=100, ll=-1.0,
                         t_range=[0, 700] if run1 else [300, 560], shuffle="aligned (A3)", onset_convention="floor(onset_float+1e-6), t_k = bin start",
                         role="diagnostic" if kind.startswith("shuf") else "analysis", groups=["pulse", "control"])
                if kind.startswith("shuf"): r["kappa"] = [float(rng.normal(0, 0.2)), float(rng.normal(0, 0.2))]
                elif kind == "s3":
                    r["groups"] = ["pulse_early", "pulse_late", "control", "washout"]; r["kappa"] = [kp + 0.05, kp - 0.05, kc, kc + 0.1]
                else: r["kappa"] = [kp, kc]
                if kind == "primary": r.update(dI_per_N=float(rng.normal(0, 0.001)), dchiJ_per_N=float(rng.normal(0, 0.01)))
                if nonconverge and nonconverge[0] == s and kind == "primary" and sub >= nonconverge[1]: r["converged"] = False
                if nan_in and nan_in[0] == s and kind == "primary" and sub == 0: r["kappa"][0] = float("nan")
                if m1_converged and m1_converged == (s, sub) and kind == "s3": r["kappa"][2] = -1.5
                if m1_nonconverged and m1_nonconverged == (s, sub) and kind == "primary": r["kappa"] = [-1.8, -1.2]; r["converged"] = False
                json.dump(r, open(os.path.join(root, "tasks", "%s_s%d_%s.json" % (s, sub, kind)), "w"))


def run(tmp, name, **kw):
    d = os.path.join(tmp, name); make(d, **kw); return A.main(d, os.path.join(d, "out"), quiet=True)


def test_all():
    tmp = tempfile.mkdtemp()
    S = run(tmp, "neg", delta=-0.4)
    assert S["S1"]["status"] == "complete" and S["S1"]["n"] == 21 and S["S1"]["outcome"] == "a" and S["S1"]["direction"] == "decrease", S["S1"]
    assert -0.5 < S["S1"]["hl"] < -0.3 and S["S1"]["p_fdr"] < 0.05 and S["S1"]["outcome_basis"].startswith("p_FDR")
    # leave-out and N>=5 rows have n = 19; cohorts 15 / 6 (complete data)
    assert S["sens_without_seen_sessions"]["n"] == 19 and S["sens_N5"]["n"] == 19 and S["cohort_2021"]["n"] == 15 and S["cohort_2022"]["n"] == 6
    S = run(tmp, "pos", delta=0.4, seed=1)
    assert S["S1"]["outcome"] == "a" and S["S1"]["direction"] == "increase" and S["S1"]["hl"] > 0.3
    S = run(tmp, "equiv", delta=0.0, between=0.08, seed=2)
    assert S["S1"]["outcome"] == "b" and S["S1"]["equivalent"] is True, S["S1"]
    S = run(tmp, "null", delta=0.0, between=0.8, seed=3)
    assert S["S1"]["outcome"] == "c" and S["S1"]["equivalent"] is False, S["S1"]
    S = run(tmp, "small", delta=0.06, between=0.01, seed=4)
    assert S["S1"]["p"] < 0.001 and abs(S["S1"]["hl"]) < 0.1 and S["S1"]["outcome"] == "b", S["S1"]            # significant but small: NOT (a)
    S = run(tmp, "both", delta=0.15, between=0.05, seed=5)
    assert S["S1"]["outcome"] == "a" and S["S1"]["bounded_within_0.3"] is True and S["S1"]["equivalent"] is True, S["S1"]   # A5: (a) wins, bounded
    # missing session files and a missing kind -> incomplete, no verdict, listed
    S = run(tmp, "missing", delta=-0.4, drop=("210421", "220517"))
    assert S["integrity"]["status"] == "incomplete" and S["S1"]["status"] == "incomplete" and S["S1"]["outcome"] is None and S["integrity"]["missing_files"] > 0
    # section-3 failure: 210606 (N=30) has only 2 of 10 converged primary subsets -> fails the kind, n=20, n_reduced, verdict with n stated
    S = run(tmp, "failed", delta=-0.4, nonconverge=("210606", 2))
    assert S["S1"]["sessions_failed_kind"] == ["210606"] and S["S1"]["n"] == 20 and S["S1"]["status"] == "n_reduced" and S["S1"]["outcome"] == "a", S["S1"]
    assert S["sens_round"]["n"] == 21                                                                          # other kinds unaffected
    # 5 converged subsets of 10 is enough (boundary of the >= 5 rule)
    S = run(tmp, "five", delta=-0.4, nonconverge=("210606", 5))
    assert S["S1"]["n"] == 21 and S["S1"]["status"] == "complete"
    # NaN kappa is unusable: the N<=16 session fails the kind; p is finite; nothing becomes a silent p_FDR = 1
    S = run(tmp, "nan", delta=-0.4, nan_in=("210421", 0))
    assert S["S1"]["sessions_failed_kind"] == ["210421"] and S["S1"]["n"] == 20 and np.isfinite(S["S1"]["p"]) and S["S1"]["p_fdr"] < 0.05, S["S1"]
    assert S["integrity"]["nonusable_fits"] == 1
    # run-1 rows abort
    try: run(tmp, "run1", delta=0.0, run1=True); raise SystemExit("run-1 rows must abort")
    except AssertionError as e: assert "pre-registered range" in str(e)
    # kappa < -1: counted on converged s3 fits (all four kappa), not on non-converged primary fits
    S = run(tmp, "m1", delta=-0.4, m1_converged=("210515", 3), m1_nonconverged=("210422", 0))
    assert S["kappa_below_minus1"]["s3"]["fits_below_minus1"] == 1 and S["kappa_below_minus1"]["s3"]["sessions"] == ["210515"]
    assert S["kappa_below_minus1"]["primary"]["fits_below_minus1"] == 0, S["kappa_below_minus1"]["primary"]
    assert S["S3_washout-control"]["sessions_with_kappa_below_minus1"] == ["210515"]
    # diagnostic cannot be quoted: no p / equivalent; mixed model records reliability fields
    assert "p" not in S["diag_null_corrected"] and "equivalent" not in S["diag_null_corrected"]
    assert {"converged", "random_effect_variance", "warnings", "ci_reliable"} <= set(S["mixed_mouse"]), S["mixed_mouse"]
    assert S["S1"]["n"] == 20 and S["S1"]["sessions_failed_kind"] == ["210422"]                                 # the non-converged primary fit fails that session
    shutil.rmtree(tmp); print("ALL ANALYSIS FIXTURE TESTS PASSED")


if __name__ == "__main__":
    test_all()
