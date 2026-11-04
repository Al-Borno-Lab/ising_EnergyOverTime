"""Regenerates the six-panel 1-ms vs 10-ms energy figure (chosen by the PI as a headline result) from the saved 1-ms fits.
    python fig_onems_headline.py DATA_DIR OUT_DIR
Per condition (onems/res_<session>.npz, rolled so index 4000 / 400 = peak velocity): f/c/fr at 1 ms (7000) and at 10 ms (700, same job).
Curves = medians over sessions of per-session means (3 conditions). Bands: acceleration (-90..0 ms) and deceleration (0..+120 ms), median durations."""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]; FOLD = os.path.join(OUT, "S3_bin_size_1ms"); os.makedirs(FOLD, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "pdf.fonttype": 42})
S = {}
for f in sorted(glob.glob(os.path.join(D, "onems/res_*.npz"))):
    z = np.load(f, allow_pickle=True); keys = sorted(set(k.split("__")[0] for k in z.files if "__f1" in k))
    S[os.path.basename(f)[4:10]] = {n: np.mean([z[f"{k}__{n}"] for k in keys], 0) for n in ("f1", "c1", "fr1", "f10", "c10", "fr10")}
med = lambda n: np.median([S[s][n] for s in S], 0)
iqr = lambda n: np.percentile([S[s][n] for s in S], [25, 75], axis=0)
t1 = np.arange(7000) - 4000.0; t10 = (np.arange(700) - 400) * 10.0
sum10 = lambda a: a.reshape(-1, 10).sum(1)                      # 1-ms values summed within each 10-ms bin
COL = dict(t="#222", f="#e0861f", c="#7b3294", r="#c0392b")
fig, ax = plt.subplots(2, 3, figsize=(20, 9.5))

def energy_panel(a, t, F, C, R, lab_r, title, win):
    w = (t >= win[0]) & (t <= win[1])
    for arr, lo_hi, col, lab in ((F + C, None, COL["t"], "total energy"), (F, None, COL["f"], "field term  −Σ h·s"), (C, None, COL["c"], "coupling term  −Σ J·s·s")):
        a.plot(t[w], arr[w], color=col, lw=1.8, label=lab)
    a.axvspan(-90, 0, color="#cfdcef", alpha=.6, lw=0); a.axvspan(0, 120, color="#f2d3d3", alpha=.6, lw=0); a.axvline(0, color="k", ls=":", lw=.7)
    a2 = a.twinx(); a2.plot(t[w], R[w], color=COL["r"], lw=1, label="population firing rate (right axis)"); a2.set_ylabel(lab_r, color=COL["r"]); a2.tick_params(axis="y", colors=COL["r"])
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
    a.set_xlabel("time from peak velocity (ms)"); a.set_title(title, loc="left")

energy_panel(ax[0, 0], t10, med("f10"), med("c10"), med("fr10"), "P(spike) per 10-ms bin", "A  10-ms bins (original)", (-1000, 2000))
ax[0, 0].set_ylabel("energy per neuron per 10-ms bin")
energy_panel(ax[0, 1], t10, sum10(med("f1")), sum10(med("c1")), sum10(med("fr1")), "expected spikes per 10 ms\n(sum of 1-ms probabilities)", "B  1-ms fit, energy summed over each 10 ms", (-1000, 2000))
ax[0, 1].set_ylabel("energy per neuron per 10 ms (sum of 10 1-ms bins)")
energy_panel(ax[0, 2], t1, med("f1"), med("c1"), med("fr1"), "P(spike) per 1-ms bin", "C  1-ms fit, per 1-ms bin", (-1000, 2000))
ax[0, 2].set_ylabel("energy per neuron per 1-ms bin")
pa_med = -40
for a, (n10, n1, lab) in zip(ax[1], (("f10", "f1", "field term"), ("c10", "c1", "coupling term"), ("fr10", "fr1", "population firing rate"))):
    out = []
    for t, n in ((t10, n10), (t1, n1)):
        m = med(n); d = m - m[(t >= -1000) & (t <= -400)].mean(); w = (t >= -400) & (t <= 800); sc = np.abs(d[w]).max(); out.append((t[w], d[w] / sc, sc))
    a.plot(out[0][0], out[0][1], color="#444", lw=2, label=f"10-ms fit (÷ {out[0][2]:.3g})")
    a.plot(out[1][0], out[1][1], color="#2f78b5", lw=.8, label=f"1-ms fit, per 1-ms bin (÷ {out[1][2]:.3g})")
    a.axvspan(-90, 0, color="#cfdcef", alpha=.6, lw=0); a.axvspan(0, 120, color="#f2d3d3", alpha=.6, lw=0)
    a.axvline(0, color="k", ls=":", lw=.7); a.axvline(pa_med, color="#2f78b5", ls="--", lw=.8, label=f"median peak acceleration ({pa_med} ms)")
    a.set_xlabel("time from peak velocity (ms)"); a.legend(frameon=False, fontsize=8)
    a.set_title(f"{'DEF'[['field term', 'coupling term', 'population firing rate'].index(lab)]}  {lab}: change from baseline (−1000…−400 ms), scaled to max |change|", loc="left")
fig.text(.01, -.01, "Time-varying fields + fixed J, pseudo-likelihood, exclusions applied, 21 sessions; knots at the same ms as the 10-ms fit (67 splines). "
         "Blue/red bands = acceleration/deceleration (median durations). Medians of per-session means.", fontsize=8, color="#555")
fig.tight_layout()
for e in ("png", "pdf"): fig.savefig(os.path.join(FOLD, f"onems_headline_6panel.{e}"), dpi=200, bbox_inches="tight")
print("saved", len(S), "sessions")
