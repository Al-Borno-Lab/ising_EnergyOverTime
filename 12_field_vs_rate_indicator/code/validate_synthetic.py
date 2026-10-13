"""E12 METHOD VALIDATION on synthetic data (not evidence about the brain).
Scenarios: dipJ0 (planted field dip, J=0), dipJ (planted dip, J>0), null (no dip, J=0).
Generates spikes from an exact N=12 Ising model with time-varying fields, then runs the same pipeline
(compute_signals + stats_e12) and reports: planted-dip recovery (T1 hits, |trough - pa|), FR failing to see it, and FT vs FT0 difference."""
import argparse, os
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import compute_signals as C, stats_e12 as S, e12_lib as L

N, R, PA, ZC = 12, 20, 397, 401

def simulate(rng, planted, Jscale, nsess=21, ncond=2):
    states = 2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1
    items = []
    for s in range(nsess):
        sess = sorted(S.MOUSE)[s % 21]
        for c in range(ncond):
            h0 = rng.normal(0.0, 0.7, N)
            amp = (0.9 * np.sign(h0) * np.abs(rng.normal(1, 0.3, N))) if planted else np.zeros(N)
            t = np.arange(L.T)
            g = np.exp(-(t - PA) ** 2 / (2 * 8.0 ** 2))
            h = h0[None] + g[:, None] * amp[None]
            J = np.zeros((N, N))
            if Jscale > 0:
                J = np.triu(rng.normal(0.0, Jscale, (N, N)), 1); J = J + J.T
            q = 0.5 * np.einsum("si,ij,sj->s", states, J, states)
            E = states @ h.T + q[:, None]                       # (2^N, T)
            P = np.exp(E - E.max(0)); P /= P.sum(0)
            cdf = np.cumsum(P, 0)
            u = rng.rand(R, L.T)
            idx = np.array([[np.searchsorted(cdf[:, tt], u[r, tt]) for tt in range(L.T)] for r in range(R)])
            idx = np.minimum(idx, 2 ** N - 1)
            X = ((states[idx] + 1) / 2).astype(np.uint8)         # (R,T,N)
            fold = np.arange(R) % 5
            items.append((f"{sess}_{c}", X, fold, PA, ZC))
    return items

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--nsess", type=int, default=21)
    a = ap.parse_args(); rng = np.random.RandomState(2026)
    res = []
    for name, planted, Js in (("dipJ0", True, 0.0), ("dipJ", True, 0.12), ("null", False, 0.0)):
        d = f"{a.out}/{name}"
        C.run(simulate(rng, planted, Js, a.nsess), d, nproc=a.nproc)
        s, summ, prim, sec = S.run(f"{d}/rows.csv", d)
        sm = summ.set_index("signal")
        u1 = prim[(prim.test == "T1") & (prim.B == "FT0")].iloc[0]; u1b = prim[(prim.test == "T2") & (prim.B == "FT0")].iloc[0]
        for sig in ("FT", "FT0", "FTu", "FT0u", "FR", "dFR", "POLb", "TRAJ", "TRAJd", "DEC"):
            res.append(dict(scenario=name, signal=sig, hit_real=sm.loc[sig, "hit_real"], hit_ctrl=sm.loc[sig, "hit_ctrl"],
                            dev_ms_median=sm.loc[sig, "dev_ms_median"], depth_ratio=sm.loc[sig, "depth_ratio_median"]))
        res.append(dict(scenario=name, signal="U1_FT_minus_FT0", hit_real=u1.mean_diff, hit_ctrl=np.nan, dev_ms_median=u1b.mean_diff,
                        depth_ratio=np.nan, U1_T1_ci=f"[{u1.ci_lo:.3f},{u1.ci_hi:.3f}] {u1.verdict_ci}", U1_T2_ci=f"[{u1b.ci_lo:.1f},{u1b.ci_hi:.1f}] ms {u1b.verdict_ci}"))
    pd.DataFrame(res).to_csv(f"{a.out}/synthetic_validation.csv", index=False)
    print(pd.DataFrame(res).to_string())
