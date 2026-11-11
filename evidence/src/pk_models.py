"""EXPLORATORY. P(K) for session 210614 stim 0: data and five model predictions (in-sample).
    python pk_models.py DATA_DIR REPO_DIR MCH_PK_NPY OUT_PNG
Data and the PL models use the 17 non-excluded reaches; the stored MCH stationary fit used all 21 reaches of the block
(its own data are drawn too, thin dotted). Model P(K) = average over time bins of P(K | t); pairwise models by Gibbs sampling."""
import sys, os
import numpy as np, pandas as pd
from scipy.special import expit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, REPO, MCH, OUT = sys.argv[1:5]
sys.path.insert(0, os.path.join(REPO, "09_claude_science_handoff/code")); import tvfield_model as tv
INK, GREY, PAIR, COUP = "#0b0b0b", "#9a9893", "#2a78d6", "#6250d6"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
Z = np.load(os.path.join(D, "spikes_v2.npz")); LM = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/phase_landmarks.csv"))
X = Z["X_210614_0"][Z["exclude_210614_0"] == 0].astype(np.uint8); R, T, N = X.shape; K = np.arange(N + 1)
zc = int(LM[(LM.session == 210614) & (LM.stim == 0)].zero_cross.iloc[0])
real = np.bincount(X.reshape(-1, N).sum(1), minlength=N + 1) / (R * T)
ind = np.array([1.0])
for p in X.reshape(-1, N).mean(0): ind = np.convolve(ind, [1 - p, p])
def pk_indep_t(h):
    P = np.zeros((h.shape[0], N + 1)); P[:, 0] = 1.0; pr = expit(2 * h)
    for i in range(N): P[:, 1:] = P[:, 1:] * (1 - pr[:, i:i + 1]) + P[:, :-1] * pr[:, i:i + 1]; P[:, 0] *= 1 - pr[:, i]
    return P.mean(0)
def gibbs(h, J, rng, chains=400, sweeps=250, burn=50):
    S = np.where(rng.random((h.shape[0], chains, N)) < expit(2 * h)[:, None, :], 1.0, -1.0); hist = np.zeros(N + 1)
    for sw in range(sweeps):
        for i in range(N): S[:, :, i] = np.where(rng.random(S.shape[:2]) < expit(2 * (h[:, None, i] + S @ J[:, i])), 1.0, -1.0)
        if sw >= burn: hist += np.bincount((S > 0).sum(2).ravel(), minlength=N + 1)
    return hist / hist.sum()
B = tv.bspline_basis(zc); m0 = tv.fit(X, B, "M0"); m1 = tv.fit(X, B, "M1")
p0 = pk_indep_t(m0["h"]); p1 = gibbs(m1["h"], m1["J"], np.random.default_rng(2))
mch = np.load(MCH); real21, pmch = mch[0], mch[1]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
lines = [(real, INK, "-", 1.6, "data (17 reaches)"), (real21, INK, ":", 1.0, "data, all 21 reaches (MCH fit data)"),
         (ind, GREY, "-", 1.0, "independent, constant rates"), (p0, GREY, "--", 1.4, "independent, time-varying rates"),
         (pmch, PAIR, "--", 1.4, "pairwise, constant fields (original MCH fit)"), (p1, PAIR, "-", 1.6, "pairwise, time-varying fields")]
for y, c, ls, lw, lab in lines:
    ok = y > 0; ax.semilogy(K[ok], y[ok], color=c, ls=ls, lw=lw, label=lab)
ax.set_ylim(1e-7, .5); ax.set_xlabel("K = cells active in the same 10-ms bin"); ax.set_ylabel("P(K)")
ax.set_title("Session 210614, baseline block (25 cells)", loc="left"); ax.legend(frameon=False, fontsize=6.8, loc="lower center")
fig.tight_layout(); fig.savefig(OUT, dpi=250, bbox_inches="tight")
for k_ in (3, 4, 5, 6, 25):
    print(k_, "data/TV-pairwise %.0f  data21/MCH %.0f  data/TV-indep %.0f" % (real[k_] / p1[k_] if p1[k_] else np.inf, real21[k_] / pmch[k_] if pmch[k_] else np.inf, real[k_] / p0[k_]))
