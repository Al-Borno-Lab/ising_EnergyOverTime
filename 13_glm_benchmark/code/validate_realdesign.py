"""E13 METHOD VALIDATION at the real design (synthetic spikes; not evidence about the brain).
For each of the 63 real conditions: real N, kept reach count R, real fold_k assignment of the kept reaches, real landmarks (onset, zc, decel_end),
and real per-neuron baseline rates (rates only, no outcomes read). Spikes are simulated:
  H1 (same-bin J constant, polarising field bump at peak acceleration), H3 (same-bin J doubled in the acceleration phase),
  H2 (kinetic lag-1 coupling), H0 (J = 0). Then the SAME pipeline as the real run: fit_glms (20 shuffles), J-constant reference (5 reps), stats_e13.
Block/condition-specific rates come from the real condition; J ~ N(0.02, 0.05) (real mean|J| about 0.02-0.2)."""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fit_glms as F, stats_e13 as S, glm_lib as G
import tvfield_model as tv


def design(npz, lm):
    z = np.load(npz, allow_pickle=True); out = []
    for _, r in lm.iterrows():
        key = f"{int(r.session)}_{int(r.stim)}"; keep = z["exclude_" + key] == 0
        X = z["X_" + key][keep]; fold = z["fold_" + key][keep]
        bl = slice(int(r.zero_cross) - 100, int(r.zero_cross) - 40)
        p = np.clip(X[:, bl, :].mean((0, 1)), 0.02, 0.98)                 # real baseline rates only
        out.append(dict(key=key, N=X.shape[2], R=X.shape[0], fold=fold, rates=p, lm=r))
    return out


def fields(d, rng, bump):
    r = d["lm"]; pa = int(r.peak_acc); N = d["N"]
    h0 = 0.5 * np.log(d["rates"] / (1 - d["rates"]))
    amp = bump * np.sign(h0 + 1e-9) * np.abs(rng.normal(1, 0.3, N)) * rng.choice([1, 1, -1], N)   # most cells polarise, some reverse
    g = np.exp(-(np.arange(G.T) - pa) ** 2 / (2 * 8.0 ** 2))
    return h0[None] + g[:, None] * amp[None]


def simulate(d, rng, scen):
    N, R = d["N"], d["R"]; r = d["lm"]
    h = fields(d, rng, 0.0 if scen == "H0" else 0.6)
    if scen == "H0": return rng.rand(R, G.T, N) < 1 / (1 + np.exp(-2 * h))[None]
    J = np.triu(rng.normal(0.02, 0.05, (N, N)), 1); J = J + J.T
    if scen == "H2":
        X = np.zeros((R, G.T, N), np.uint8); prev = 2.0 * (rng.rand(R, N) < 0.5) - 1
        for t in range(G.T):
            p = 1 / (1 + np.exp(-2 * (h[t][None] + prev @ (3 * J)))); x = (rng.rand(R, N) < p); X[:, t] = x; prev = 2.0 * x - 1
        return X
    if scen == "H3":   # J doubled during the acceleration phase: implement by sampling twice and splicing the acceleration bins
        Xa = G.sample_m1(h, J, R, rng); Xb = G.sample_m1(h, 2 * J, R, rng)
        Xa[:, int(r.onset):int(r.zero_cross)] = Xb[:, int(r.onset):int(r.zero_cross)]; return Xa
    return G.sample_m1(h, J, R, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--landmarks", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nproc", type=int, default=12); ap.add_argument("--scen", default="H1,H3,H2,H0")
    ap.add_argument("--nshuf", type=int, default=20)
    a = ap.parse_args(); lm = pd.read_csv(a.landmarks); D = design(a.npz, lm); rng = np.random.RandomState(4242); rows = []
    for scen in a.scen.split(","):
        d_out = f"{a.out}/{scen}"; os.makedirs(d_out, exist_ok=True)
        jobs = [(d["key"], simulate(d, rng, scen).astype(np.uint8), d["fold"], int(d["lm"].zero_cross), a.nshuf, d_out, G.MODELS_ALL, G.LAMJ) for d in D]
        F.run(jobs, a.nproc)
        F.run_ref([(k, X, f, zc, G.PREREG["n_ref_reps"], d_out) for k, X, f, zc, *_ in jobs], a.nproc)
        ct, s, fam, g1 = S.run(d_out, a.landmarks, d_out)
        A = s[s.phase == "all"]
        row = dict(scenario=scen, G1_est=g1["est"], G1_ci=f"[{g1['ci_lo']:.4f},{g1['ci_hi']:.4f}]", G1_equiv=g1["equivalent_pm0.01"], gain_SYNC=A.gain_SYNC.median(),
                   gain_LAG1=A.gain_LAG1.median(), gain_LAG1H=A.gain_LAG1H.median(), net_SYNC=A.net_SYNC.median(), net_LAG1H=A.net_LAG1H.median())
        for _, q in fam.iterrows():
            ph = next((p for p in ("acceleration", "deceleration", "post") if p in q.test), "")
            row[q.test.split(":")[0] + ("_" + ph if ph else "")] = f"{q.est:.4f} [{q.ci_lo:.4f},{q.ci_hi:.4f}] p={q.p:.3f} p_bh={q.p_bh:.3f}"
        rows.append(row); print(pd.DataFrame([row]).T.to_string(), flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/realdesign_validation.csv", index=False)
