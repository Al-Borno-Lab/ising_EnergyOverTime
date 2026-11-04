"""EXPLORATORY. Reconciliation with Calame et al. (2023): their pooled Purkinje-cell rate (all cells across animals) drops sharply at
peak acceleration. Do we reproduce that drop, and does the field-term dip survive after the rate is regressed out?
    python calame_check.py DATA_DIR REPO_DIR OUT_DIR
(a) Pooled rate change (Hz) over all cell x condition PSTHs, aligned to each condition's peak acceleration (Calame-style pooling;
    not velocity-binned: our kinematics are min-max scaled per session).
(b) Per session: rate change and field-term change in the peak-acceleration window [-100, +50] ms vs the window's own early baseline.
(c) Field term with the population rate regressed out (per condition, ordinary least squares over -600..+1000 ms on baseline-z traces
    from the E12 pipeline, regen 'ovl'): does the residual still dip at peak acceleration?"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FOLD = os.path.join(OUT, "4_polarization_at_peak_acceleration"); os.makedirs(FOLD, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
PRE, POST = 60, 100                       # bins around peak acceleration (10 ms)
W = (PRE - 10, PRE + 5)                    # [-100, +50] ms window (inclusive bin indices into the 161-bin trace)
EARLY = (0, 19)                            # -600..-410 ms: early reference inside the trace
t = (np.arange(-PRE, POST + 1)) * 10
Z = dict(np.load(os.path.join(D, "spikes_v2.npz"))); LM = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/phase_landmarks.csv"))

cells, sess_rate = [], {}
for k in sorted(x[2:] for x in Z if x.startswith("X_")):
    s, st = map(int, k.split("_")); r = LM[(LM.session == s) & (LM.stim == st)].iloc[0]; pa, zc = int(r.peak_acc), int(r.zero_cross)
    X = Z["X_" + k][Z["exclude_" + k] == 0].astype(float); psth = X.mean(0) * 100.0          # Hz-equivalent (spike prob per 10 ms x 100)
    base = psth[zc - 100:zc - 40].mean(0); d = psth[pa - PRE:pa + POST + 1] - base             # (161, N)
    cells.append(d); sess_rate.setdefault(s, []).append(d.mean(1))
cells = np.concatenate(cells, 1); pooled = cells.mean(1)
ses = sorted(sess_rate); R = np.array([np.mean(sess_rate[s], 0) for s in ses])                  # session-mean rate change (21, 161)
rng = np.random.default_rng(0); bs = np.array([R[rng.integers(0, len(R), len(R))].mean(0) for _ in range(2000)])

O, keys = [], []
for f in sorted(glob.glob(os.path.join(D, "regen/regen_*.npz"))):
    z = np.load(f); O.append(z["ovl"]); keys += [str(x) for x in z["keys"]]
O = np.concatenate(O); FT, FR = O[:, 0], O[:, 3]
resid = np.empty_like(FT)
for i in range(len(O)):
    A = np.column_stack([np.ones(len(t)), FR[i]]); resid[i] = FT[i] - A @ np.linalg.lstsq(A, FT[i], rcond=None)[0]
dip = lambda M: M[:, W[0]:W[1] + 1].mean(1) - M[:, EARLY[0]:EARLY[1] + 1].mean(1)
cs = pd.DataFrame(dict(key=keys, session=[int(k.split("_")[0]) for k in keys], FT=dip(FT), FR=dip(FR), FT_resid=dip(resid)))
S = cs.groupby("session")[["FT", "FR", "FT_resid"]].mean()
S["rate_Hz"] = pd.Series(dip(R), index=ses)
stats = {c: dict(median=float(S[c].median()), n_neg=int((S[c] < 0).sum()), n=int(S[c].notna().sum()), p=float(wilcoxon(S[c].dropna()).pvalue)) for c in S}
stats["pooled_rate_trough_ms_from_peak_acc"] = int(t[pooled.argmin()]); stats["pooled_rate_trough_Hz"] = round(float(pooled.min()), 2)
stats["pooled_rate_window_change_Hz"] = round(float(pooled[W[0]:W[1] + 1].mean() - pooled[EARLY[0]:EARLY[1] + 1].mean()), 2)
stats["n_cell_conditions"] = int(cells.shape[1]); stats["median_r_FT_vs_FR"] = float(np.median([np.corrcoef(FT[i], FR[i])[0, 1] for i in range(len(O))]))
S.reset_index().to_csv(os.path.join(FOLD, "calame_check_by_session.csv"), index=False)
json.dump(stats, open(os.path.join(FOLD, "calame_check_stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a = ax[0]; a.axvspan(t[W[0]], t[W[1]], color="#f2d3d3", lw=0)
a.fill_between(t, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color="#c0392b", alpha=.15, lw=0, label="session bootstrap 95%")
a.plot(t, pooled, color="#c0392b", lw=2, label=f"pooled over {cells.shape[1]} cell-conditions (Calame-style)")
a.axhline(0, color="#999", lw=.6); a.axvline(0, color="k", ls=":", lw=.7); a.set_xlabel("time from peak acceleration (ms)"); a.set_ylabel("rate change from baseline (spikes/s)")
a.set_title("a  Pooled rate: dip near peak acceleration", loc="left"); a.legend(frameon=False, fontsize=7.5)
a = ax[1]
for i, (c, lab, col) in enumerate((("FT", "field term", "#1f4e99"), ("FR", "mean rate", "#c0392b"), ("FT_resid", "field term,\nrate regressed out", "#2a9d8f"))):
    v = S[c].values; a.scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=14, color=col, alpha=.75); a.plot([i - .25, i + .25], [np.median(v)] * 2, color="k", lw=2)
    a.text(i, 1.01, f"{stats[c]['n_neg']}/21 below 0", ha="center", va="bottom", fontsize=8, transform=a.get_xaxis_transform())
a.axhline(0, color="#999", lw=.6); a.set_xticks(range(3), ["field term", "mean rate", "field term,\nrate removed"])
a.set_ylabel("change in [-100, +50] ms window\n(baseline z, session mean)"); a.set_title("b  Per session: dip at peak acceleration", loc="left", pad=16)
a = ax[2]
for M, lab, col in ((FT, "field term", "#1f4e99"), (resid, "field term, rate regressed out", "#2a9d8f"), (FR, "mean rate", "#c0392b")):
    m, se = M.mean(0), M.std(0) / np.sqrt(len(M)); a.plot(t, m, color=col, lw=2, label=lab); a.fill_between(t, m - se, m + se, color=col, alpha=.15, lw=0)
a.axvspan(t[W[0]], t[W[1]], color="#f2d3d3", lw=0, zorder=0); a.axvline(0, color="k", ls=":", lw=.7); a.axhline(0, color="#999", lw=.6)
a.set_xlabel("time from peak acceleration (ms)"); a.set_ylabel("baseline z (mean ± s.e.m., 63 conditions)"); a.legend(frameon=False, fontsize=7.5); a.set_title("c  The dip is not the rate", loc="left")
fig.text(.01, -.03, "EXPLORATORY. Pooled rate = mean over all cells of all 63 conditions (PSTH minus each cell's baseline, -1000..-410 ms from peak velocity); not velocity-binned. "
         "Field term and rate traces: E12 pipeline (regen). Shaded = [-100, +50] ms test window.", fontsize=7.5, color="#555")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(FOLD, f"calame_reconciliation.{e}"), dpi=200, bbox_inches="tight")
