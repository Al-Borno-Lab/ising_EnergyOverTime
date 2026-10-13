"""E13 METHOD VALIDATION on synthetic data (not evidence about the brain).
S1 sync: exact Ising, same-bin J constant, bins independent in time given t.  Expect: SYNC == M1 (G1), LAG1/LAG1H net gain ~ 0, G3a ~ 0.
S2 kinetic: spike at t depends on spins at t-1 (positive control for the lag models). Expect: LAG1 gain > 0.
S3 epochJ: as S1 but J doubled during the acceleration window (power check for G3a)."""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fit_glms as F, stats_e13 as S, glm_lib as G

N, R, T = 12, 20, 700
LM = dict(onset=391, zero_cross=401, decel_end=416)
ZC = LM["zero_cross"]


def fields(rng):
    h0 = rng.normal(0, 0.7, N); amp = 0.9 * np.sign(h0) * np.abs(rng.normal(1, 0.3, N))
    t = np.arange(T); g = np.exp(-(t - 397) ** 2 / (2 * 8.0 ** 2))
    return h0[None] + g[:, None] * amp[None]


def sim_sync(rng, jscale_t):
    states = 2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) - 1
    h = fields(rng); J = np.triu(rng.normal(0.05, 0.12, (N, N)), 1); J = J + J.T
    q = 0.5 * np.einsum("si,ij,sj->s", states, J, states)
    E = states @ h.T + q[:, None] * jscale_t[None]
    P = np.exp(E - E.max(0)); P /= P.sum(0); cdf = np.cumsum(P, 0)
    u = rng.rand(R, T)
    idx = np.minimum(np.array([[np.searchsorted(cdf[:, t], u[r, t]) for t in range(T)] for r in range(R)]), 2 ** N - 1)
    return ((states[idx] + 1) / 2).astype(np.uint8)


def sim_kinetic(rng):
    h = fields(rng) * 0.6; J = np.triu(rng.normal(0.0, 0.5, (N, N)), 1); J = J + J.T
    X = np.zeros((R, T, N), np.uint8); prev = 2.0 * (rng.rand(R, N) < 0.5) - 1
    for t in range(T):
        p = 1 / (1 + np.exp(-2 * (h[t][None] + prev @ J)))
        x = (rng.rand(R, N) < p).astype(np.uint8); X[:, t] = x; prev = 2.0 * x - 1
    return X


def make(rng, scen, nsess):
    items = []
    for s in range(nsess):
        sess = sorted(S.MOUSE)[s % 21]
        if scen == "S1": X = sim_sync(rng, np.ones(T))
        elif scen == "S3": sc = np.ones(T); sc[LM["onset"]:ZC] = 2.0; X = sim_sync(rng, sc)
        else: X = sim_kinetic(rng)
        items.append((f"{sess}_0", X, np.arange(R) % 5, ZC))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--nsess", type=int, default=21); ap.add_argument("--nshuf", type=int, default=20)
    a = ap.parse_args(); rng = np.random.RandomState(1313)
    lm = pd.DataFrame([dict(session=s, stim=0, **LM) for s in sorted(S.MOUSE)])
    lm.to_csv(f"{a.out}/synthetic_landmarks.csv", index=False) if os.makedirs(a.out, exist_ok=True) is None else None
    summ = []
    for scen in ("S1", "S2", "S3"):
        d = f"{a.out}/{scen}"; os.makedirs(d, exist_ok=True)
        jobs = [(k, X, f, zc, a.nshuf, d, G.MODELS_ALL, G.LAMJ) for k, X, f, zc in make(rng, scen, a.nsess)]
        F.run(jobs, a.nproc)
        F.run_ref([(k, X, f, zc, 5, d) for k, X, f, zc, *_ in jobs], a.nproc)
        ct, s, fam, g1 = S.run(d, f"{a.out}/synthetic_landmarks.csv", d)
        A = s[s.phase == "all"]
        row = dict(scenario=scen, G1_mean_diff=g1["est"], G1_ci=f"[{g1['ci_lo']:.4f},{g1['ci_hi']:.4f}]", G1_equiv=g1["equivalent_pm0.01"],
                   gain_SYNC=A.gain_SYNC.median(), gain_M1=A.gain_M1.median(), gain_LAG1=A.gain_LAG1.median(), gain_LAG1H=A.gain_LAG1H.median(),
                   net_SYNC=A.net_SYNC.median(), net_LAG1H=A.net_LAG1H.median())
        for _, r in fam.iterrows():
            ph = next((p for p in ("acceleration", "deceleration", "post") if p in r.test), "")
            row[r.test.split(":")[0] + ("_" + ph if ph else "")] = f"{r.est:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}] p={r.p:.3f}"
        summ.append(row)
        print(pd.DataFrame([row]).T.to_string(), flush=True)
    pd.DataFrame(summ).to_csv(f"{a.out}/synthetic_validation.csv", index=False)
