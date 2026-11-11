"""Paper figures (Results Figs 1-5, Supplementary S1-S2) in one visual style: one palette with a fixed meaning per colour,
capital panel letters, no dual axes (firing rate gets its own panel). Rebuilt from saved tables and data only.
    python paper_figures.py DATA_DIR REPO_DIR E10_DIR E14_DIR E15_DIR OUT_DIR
Colour meaning (validated for colour-vision deficiency per co-occurring set; see the dataviz palette check):
  field term / drive = orange, coupling term = violet, pairwise model = blue, firing rate = aqua,
  data and model-free signals = ink, independent model / controls = greys."""
import sys, os, glob
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D, REPO, E10, E14, E15, OUT = sys.argv[1:7]; OUT = os.path.join(OUT, "paper"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "09_claude_science_handoff/code")); import tvfield_model as tv

INK, INK2, GREY, LGREY = "#0b0b0b", "#52514e", "#9a9893", "#cfcec9"
FIELD, COUP, PAIR, RATE = "#eb6834", "#6250d6", "#2a78d6", "#1baf7a"
ACC, DEC = "#dfe9f7", "#f8e3e3"
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2, "axes.linewidth": .6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": .6, "ytick.major.width": .6, "lines.linewidth": 1.4,
                     "legend.frameon": False, "legend.fontsize": 6.8, "pdf.fonttype": 42, "axes.titlelocation": "left"})
LM = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/phase_landmarks.csv"))
Z = dict(np.load(os.path.join(D, "spikes_v2.npz")))

def letter(ax, L, x=-0.2): ax.text(x, 1.04, L, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
def bands(ax, a=(-90, 0), d=(0, 120)):
    ax.axvspan(*a, color=ACC, lw=0, zorder=0); ax.axvspan(*d, color=DEC, lw=0, zorder=0); ax.axvline(0, color=INK2, lw=.6, ls=":", zorder=1)
def zero(ax): ax.axhline(0, color=LGREY, lw=.7, zorder=0)
def dots(ax, x, v, color, hollow=False, s=9, seed=0, bar=True, w=.28):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; jx = x + np.random.default_rng(seed).uniform(-.12, .12, len(v))
    ax.scatter(jx, v, s=s, facecolor="white" if hollow else color, edgecolor=color, lw=.8, zorder=3)
    if bar: ax.plot([x - w, x + w], [np.median(v)] * 2, color=INK, lw=1.6, zorder=4)
def save(fig, name):
    for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"{name}.{e}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved", name)
def cond(key):
    s, st = map(int, key.split("_")); m = Z["exclude_" + key] == 0; r = LM[(LM.session == s) & (LM.stim == st)].iloc[0]
    return Z["X_" + key][m].astype(np.uint8), int(r.zero_cross), int(r.peak_acc)

# ============================================================== Figure 1: model and validation
def fig1():
    X, zc, pa = cond("210614_0"); m1 = tv.fit(X, tv.bspline_basis(zc), "M1"); h, J = m1["h"], m1["J"]
    w = np.arange(zc - 100, zc + 151); t = (w - zc) * 10
    dh = h[zc - 10:zc + 5].mean(0) - h[zc - 100:zc - 40].mean(0); order = np.argsort(dh)
    fig = plt.figure(figsize=(7.2, 4.6)); gs = GridSpec(2, 3, figure=fig, hspace=.55, wspace=.45)
    a = fig.add_subplot(gs[0, 0]); letter(a, "A")
    z1 = np.load(os.path.join(D, "e14_spk/210614.npz")); m1ms = (z1["stim"] == 0) & (z1["exclude"] == 0); c1 = z1["c1"][m1ms][3]   # same reach, 1-ms spikes
    tt = np.arange(c1.shape[0]) - zc * 10
    for row, i in enumerate(order):
        sp = tt[(c1[:, i] > 0) & (tt >= -1000) & (tt <= 1500)]; a.vlines(sp, row + .1, row + .9, color=INK, lw=.35)
    a.set_xlim(-1000, 1500); a.set_ylim(X.shape[2], 0); a.axvline(0, color=FIELD, lw=.8, ls=":")
    a.set_xlabel("time from peak velocity (ms)"); a.set_ylabel("cell (sorted by drive change)"); a.set_title("Spikes, one reach")
    b = fig.add_subplot(gs[0, 1]); letter(b, "B"); bands(b)
    for i in list(order[:3]) + list(order[-3:]):
        b.plot(t, h[w, i], color=FIELD, lw=1.1, ls="-" if dh[i] > 0 else "--")
    b.set_xlabel("time from peak velocity (ms)"); b.set_ylabel("field h_i(t)"); b.set_title("Drive to 6 cells")
    c = fig.add_subplot(gs[0, 2]); letter(c, "C"); vm = np.abs(J).max()
    im = c.imshow(J[np.ix_(order, order)], cmap="RdBu_r", vmin=-vm, vmax=vm); c.set_xticks([]); c.set_yticks([])
    c.set_title("Couplings J (fixed)"); cb = fig.colorbar(im, ax=c, fraction=.046, pad=.04); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    for sp in c.spines.values(): sp.set_visible(False)
    v = pd.read_csv(os.path.join(REPO, "07_presubmission_checks/timevarying_field/tvfield_validation.csv")).set_index("case")
    d = fig.add_subplot(gs[1, 0:3]); letter(d, "D", x=-.06); zero(d)
    cases = [("J true", "simulated with\ncouplings"), ("J=0", "simulated without\ncouplings"), ("J=0 + per-reach gain", "without couplings,\nshared reach-to-reach gain")]
    for k, (cid, lab) in enumerate(cases):
        for j, (col, c_, nm) in enumerate((("dPL_M1_M0", PAIR, "couplings"), ("dPL_Mlag_M0", GREY, "couplings, 200-ms lag"), ("dPL_shuffle_null", LGREY, "reach-shuffle control"))):
            d.bar(k + (j - 1) * .26, v.loc[cid, col], width=.24, color=c_, label=nm if k == 0 else None)
    d.set_xticks(range(3), [c[1] for c in cases]); d.set_ylabel("held-out gain over\nindependent model (nats/bin)")
    d.legend(loc="upper right"); d.set_title("Gain from couplings only when couplings are real (simulated data)")
    save(fig, "Fig1_model_validation")

def figS3():
    """Validation of the multi-information measure on simulated data (moved out of Fig. 1)."""
    h0 = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/h0_results.csv"))
    fig, e = plt.subplots(figsize=(2.8, 2.6)); r_ = h0[h0.scen != "C"]; c_ = h0[h0.scen == "C"]
    vals = np.r_[h0.true_dI, h0.est_dI]; pad = .1 * np.ptp(vals); lim = [vals.min() - pad, vals.max() + pad]; e.plot(lim, lim, color=LGREY, lw=.8, ls="--")
    e.scatter(r_.true_dI, r_.est_dI, s=8, color=INK); e.scatter(c_.true_dI, c_.est_dI, s=12, facecolor="white", edgecolor=GREY, lw=.8)
    e.set_xlim(lim); e.set_ylim(lim); e.set_xlabel("true change in multi-information\n(bits per neuron)"); e.set_ylabel("estimated change")
    e.set_title("Multi-information measure (simulated)")
    save(fig, "FigS3_multiinformation_validation")

# ============================================================== Figure 2: coordination beyond shared rates
def fig2():
    X, zc, pa = cond("210614_0"); Xf = X.reshape(-1, X.shape[2]).astype(int); N = Xf.shape[1]; K = np.arange(N + 1)
    pk = lambda M: np.bincount(M.sum(1), minlength=N + 1) / len(M)
    real = pk(Xf); ind = np.array([1.0])
    for p in Xf.mean(0): ind = np.convolve(ind, [1 - p, p])
    rng = np.random.default_rng(0); sh = np.array([pk(np.column_stack([rng.permutation(Xf[:, i]) for i in range(N)])) for _ in range(200)])
    lo, hi = np.percentile(sh - ind, [2.5, 97.5], 0)
    fig = plt.figure(figsize=(7.2, 4.6)); gs = GridSpec(2, 2, figure=fig, hspace=.6, wspace=.35)
    a = fig.add_subplot(gs[0, :]); letter(a, "A", x=-.07)
    slo, shi = np.percentile(sh, [2.5, 97.5], 0)
    a.fill_between(K, np.maximum(slo, 1e-7), np.maximum(shi, 1e-7), color=LGREY, lw=0, zorder=0)
    a.fill_between(K, np.maximum(ind, 1e-7), np.maximum(real, 1e-7), where=real > shi, color=COUP, alpha=.25, lw=0, interpolate=True)
    a.semilogy(K, ind, color=GREY, lw=1.4); a.semilogy(K, real, color=INK, lw=1.4)
    a.set_ylim(1e-6, .5); a.set_xlabel("K = cells active in the same 10-ms bin"); a.set_ylabel("P(K)")
    a.set_title("Population counts in one session (25 cells)")
    cv = pd.read_csv(os.path.join(REPO, "07_presubmission_checks/timevarying_field/tvfield_cv_scores.csv"))
    ph = [("baseline", "baseline"), ("acceleration", "acceleration"), ("deceleration", "deceleration"), ("post", "after reach")]
    S = cv.groupby("session").mean(numeric_only=True)
    c = fig.add_subplot(gs[1, 0]); letter(c, "B"); zero(c)
    for i, (p, lab) in enumerate(ph):
        dots(c, i * 3, S[f"gain_{p}"], PAIR, seed=i); dots(c, i * 3 + 1, S[f"lag_{p}"], GREY, seed=10 + i); dots(c, i * 3 + 2, S[f"nullp95_{p}"], LGREY, seed=20 + i)
    c.set_yscale("symlog", linthresh=.5, linscale=1.5); c.set_ylim(-.4, 9)
    c.set_xticks([i * 3 + 1 for i in range(4)], [l for _, l in ph]); c.set_ylabel("held-out gain from couplings\n(nats/bin; log above 0.5)")
    c.set_title("Gain over time-varying rates, by phase (21 sessions)")
    P = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "entropy/per_reach_surprise_*.csv"))])
    d = fig.add_subplot(gs[1, 1]); letter(d, "C"); zero(d)
    for i, (wn, lab) in enumerate((("rest", "rest"), ("reach", "reach"), ("trial", "whole trial"))):
        g = (P[f"sur_M0_{wn}"] - P[f"sur_M1_{wn}"]).groupby(P.session).mean(); dots(d, i, g, PAIR, seed=i)
    d.set_xticks(range(3), ["rest", "reach", "whole trial"]); d.set_ylabel("exact held-out gain\n(bits per neuron per bin)"); d.set_title("Exact likelihood, held-out reaches (one dot per session)")
    save(fig, "Fig2_coordination_beyond_rates")

# ============================================================== Figure 3: polarization at peak acceleration
def fig3():
    m = pd.read_csv(os.path.join(REPO, "12_field_vs_rate_indicator/assets/tables/primary_session_metrics.csv"))
    sig = [("FT", "field term,\npairwise", FIELD, False), ("FT0", "field term,\nindependent", FIELD, True), ("POLb", "polarization\nindex", INK, False),
           ("FR", "mean rate", RATE, False), ("DEC", "decoder\n(given target)", GREY, False)]
    fig = plt.figure(figsize=(7.2, 6.4)); gs = GridSpec(3, 2, figure=fig, hspace=.75, wspace=.32)
    a = fig.add_subplot(gs[0, 0]); letter(a, "A")
    def mci(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]; b = [v[np.random.default_rng(k).integers(0, len(v), len(v))].mean() for k in range(2000)]
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    for i, (s, lab, col, hol) in enumerate(sig):
        x = m[m.signal == s]
        for j, (vals, c_, h_) in enumerate(((x.hit, col, hol), (x.hit_c, LGREY, False))):
            mu, l_, u_ = mci(vals); xx = i * 2.6 + j
            a.bar(xx, mu, width=.85, color="white" if h_ else c_, edgecolor=c_, lw=1.2)
            a.errorbar(xx, mu, yerr=[[mu - l_], [u_ - mu]], color=INK, lw=.9, capsize=2)
    a.set_xticks([i * 2.6 + .5 for i in range(5)], [l for _, l, _, _ in sig], fontsize=6.3); a.set_ylim(0, 1)
    a.set_ylabel("fraction with minimum in the\n200 ms before peak velocity"); a.set_title("Timing: reach (colour) vs rest (grey); mean, 95% CI")
    b = fig.add_subplot(gs[0, 1]); letter(b, "B")
    for i, (s, lab, col, hol) in enumerate(sig):
        dots(b, i, m[m.signal == s].dev_ms, col, hollow=hol, seed=i)
    b.set_xticks(range(5), [l for _, l, _, _ in sig], fontsize=6.3); b.set_ylabel("distance of minimum from\npeak acceleration (ms)"); b.set_title("Precision (lower is better)")
    O = np.concatenate([np.load(f)["ovl"] for f in sorted(glob.glob(os.path.join(D, "regen/regen_*.npz")))]); t = np.arange(-60, 101) * 10
    c = fig.add_subplot(gs[1, 0]); letter(c, "C"); zero(c)
    for j, (lab, col, ls, lw) in enumerate((("field term, pairwise", FIELD, "-", 2.2), ("field term, independent", FIELD, (0, (2.5, 1.5)), 1.1), ("polarization index", INK, (0, (1, 1)), 1.3), ("mean rate", RATE, "-", 1.6))):
        M = O[:, j]; mu, se = M.mean(0), M.std(0) / np.sqrt(len(M)); c.plot(t, mu, color=col, ls=ls, lw=lw); c.fill_between(t, mu - se, mu + se, color=col, alpha=.10, lw=0)
    c.axvline(0, color=INK2, lw=.6, ls=":"); c.set_xlabel("time from peak acceleration (ms)"); c.set_ylabel("change (baseline SD)")
    c.set_title("Signals aligned to peak acceleration (63 conditions)")
    d = fig.add_subplot(gs[1, 1]); letter(d, "D"); zero(d)
    for i, (j, col, hol) in enumerate(((1, FIELD, True), (2, INK, False), (3, RATE, False))):
        r = [np.corrcoef(O[k, 0], O[k, j])[0, 1] for k in range(len(O))]; dots(d, i, r, col, hollow=hol, seed=i)
    d.set_xticks(range(3), ["field term,\nindependent", "polarization\nindex", "mean rate"]); d.set_ylim(-1.05, 1.05)
    d.set_ylabel("correlation with pairwise\nfield term (per condition)"); d.set_title("The dip is polarization, not mean rate")
    # pooled rate (Calame-style) and per-session window change
    cells, sess_rate = [], {}
    for k in sorted(x[2:] for x in Z if x.startswith("X_")):
        X, zc, pa = cond(k); psth = X.astype(float).mean(0) * 100; base = psth[zc - 100:zc - 40].mean(0); dd = psth[pa - 60:pa + 101] - base
        cells.append(dd); sess_rate.setdefault(int(k[:6]), []).append(dd.mean(1))
    cells = np.concatenate(cells, 1); pooled = cells.mean(1); ses = sorted(sess_rate); Rr = np.array([np.mean(sess_rate[s], 0) for s in ses])
    bs = np.array([Rr[np.random.default_rng(i).integers(0, len(Rr), len(Rr))].mean(0) for i in range(2000)])
    e = fig.add_subplot(gs[2, 0]); letter(e, "E"); zero(e)
    e.fill_between(t, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color=RATE, alpha=.18, lw=0); e.plot(t, pooled, color=RATE, lw=1.4)
    e.axvline(0, color=INK2, lw=.6, ls=":"); e.set_xlabel("time from peak acceleration (ms)"); e.set_ylabel("rate change (spikes/s)")
    e.set_title(f"Rate pooled over {cells.shape[1]:,} cells (95% session bootstrap)")
    keys = [];
    for f in sorted(glob.glob(os.path.join(D, "regen/regen_*.npz"))): keys += [int(str(k)[:6]) for k in np.load(f)["keys"]]
    keys = np.array(keys); W = slice(50, 66); E0 = slice(0, 20)
    resid = np.array([O[i, 0] - np.column_stack([np.ones(len(t)), O[i, 3]]) @ np.linalg.lstsq(np.column_stack([np.ones(len(t)), O[i, 3]]), O[i, 0], rcond=None)[0] for i in range(len(O))])
    dip = lambda M: M[:, W].mean(1) - M[:, E0].mean(1)
    f = fig.add_subplot(gs[2, 1]); letter(f, "F"); zero(f)
    for i, (vals, col, hol) in enumerate(((dip(O[:, 0]), FIELD, False), (dip(O[:, 3]), RATE, False), (dip(resid), FIELD, True))):
        dots(f, i, pd.Series(vals).groupby(keys).mean(), col, hollow=hol, seed=i)
    f.set_xticks(range(3), ["field term", "mean rate", "field term,\nrate regressed out"]); f.set_ylabel("change at peak acc.\n(baseline SD)")
    f.set_title("Per session: field term dips, rate does not reliably")
    save(fig, "Fig3_polarization")

# ============================================================== Figure 4: 10 ms vs 1 ms
def fig4():
    S = {}
    for f in sorted(glob.glob(os.path.join(D, "onems/res_*.npz"))):
        z = np.load(f, allow_pickle=True); ks = sorted(set(k.split("__")[0] for k in z.files if "__f1" in k))
        S[f[-10:-4]] = {n: np.mean([z[f"{k}__{n}"] for k in ks], 0) for n in ("f1", "c1", "fr1", "f10", "c10", "fr10")}
    med = lambda n: np.median([S[s][n] for s in S], 0); t1 = np.arange(7000) - 4000.0; t10 = (np.arange(700) - 400) * 10.0
    sm = lambda a: gaussian_filter1d(a, 5)
    fig = plt.figure(figsize=(7.2, 6.2)); gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1.6], hspace=.55, wspace=.45)
    for col, (t, f_, c_, r_, lab, L) in enumerate(((t10, med("f10"), med("c10"), med("fr10"), "10-ms bins", "AB"), (t1, sm(med("f1")), sm(med("c1")), sm(med("fr1")), "1-ms bins (5-ms smoothing)", "CD"))):
        w = (t >= -1000) & (t <= 1500)
        a = fig.add_subplot(gs[0, col]); letter(a, L[0]); bands(a)
        for arr, cl, nm in ((f_ + c_, INK, "total"), (f_, FIELD, "field term"), (c_, COUP, "coupling term")): a.plot(t[w], arr[w], color=cl, lw=1.2, label=nm)
        a.set_ylabel("energy per neuron per bin"); a.set_title(f"Energy, {lab}")
        b = fig.add_subplot(gs[1, col], sharex=a); letter(b, L[1]); bands(b); b.plot(t[w], r_[w], color=RATE, lw=1.1)
        b.set_ylabel("P(spike) per bin"); b.set_xlabel("time from peak velocity (ms)"); b.set_title(f"Population firing rate, {lab}")
    ax0 = fig.add_subplot(gs[0:2, 2]); ax0.axis("off")
    ax0.text(0, .95, "Colour key", fontweight="bold", fontsize=7.5, va="top")
    for i, (cl, nm) in enumerate(((INK, "total energy"), (FIELD, "field term (drive)"), (COUP, "coupling term"), (RATE, "firing rate"), (ACC, "acceleration phase"), (DEC, "deceleration phase"))):
        ax0.add_patch(plt.Rectangle((0, .82 - i * .1), .08, .05, color=cl, transform=ax0.transAxes)); ax0.text(.12, .845 - i * .1, nm, va="center", fontsize=7, transform=ax0.transAxes)
    ax0.text(0, .2, "E–G: thick black = 10-ms fit;\nthin colour = 1-ms fit", fontsize=7, va="top", color=INK2, transform=ax0.transAxes)
    for col, (n10, n1, lab, cl, L) in enumerate((("f10", "f1", "field term", FIELD, "E"), ("c10", "c1", "coupling term", COUP, "F"), ("fr10", "fr1", "firing rate", RATE, "G"))):
        a = fig.add_subplot(gs[2, col]); letter(a, L); bands(a); zero(a)
        for t, n, lw, ls, nm in ((t10, n10, 1.6, "-", "10 ms"), (t1, n1, .7, "-", "1 ms")):
            mm = med(n) if n.endswith("10") else sm(med(n)); dd = mm - mm[(t >= -1000) & (t <= -400)].mean(); w = (t >= -400) & (t <= 800)
            a.plot(t[w], dd[w] / np.abs(dd[w]).max(), color=INK if n.endswith("10") else cl, lw=lw, ls=ls, label=nm)
        a.set_title(lab); a.set_xlabel("time from peak velocity (ms)")
        if col == 0: a.set_ylabel("change from baseline\n/ max |change|")
    save(fig, "Fig4_resolution_1ms")

# ============================================================== Figure 5: what fixed couplings do
def fig5():
    se = pd.read_csv(os.path.join(E10, "assets/tables/e10_sessions.csv")); co = pd.read_csv(os.path.join(E10, "assets/tables/e10_conditions.csv"))
    fig = plt.figure(figsize=(7.2, 4.6)); gs = GridSpec(2, 3, figure=fig, hspace=.6, wspace=.45)
    a = fig.add_subplot(gs[0, 0]); letter(a, "A"); zero(a); a.axvline(0, color=LGREY, lw=.7)
    for coh, hol in ((2021, False), (2022, True)):
        x = se[se.cohort == coh]; a.scatter(x.real_dv, x.real_dI, s=14, facecolor="white" if hol else INK, edgecolor=INK, lw=.8, label=f"{coh} cohort")
    a.set_xlabel("change in variance p(1−p)\n(reach − rest)"); a.set_ylabel("change in multi-information\n(bits per neuron)"); a.legend(loc="lower right")
    a.set_title("Polarization does not raise\ncoupling influence")
    b = fig.add_subplot(gs[0, 1]); letter(b, "B"); zero(b); b.axvline(0, color=LGREY, lw=.7)
    b.scatter(co.real_dchiJ, co.emp_dcov, s=9, color=INK); b.set_xlabel("predicted by fixed couplings"); b.set_ylabel("observed change in covariance")
    b.set_title("Predicted vs observed\ncovariance change (63 conditions)")
    tc = {}
    for f in sorted(glob.glob(os.path.join(D, "e15/real/e15_tc_*.npz"))):
        z = np.load(f); tc.update({k: z[k] for k in z.files})
    keys = sorted(set(k.split("__")[0] for k in tc)); off = tc[f"{keys[0]}__offset"] * 10.0; ses = sorted(set(k[:6] for k in keys))
    c = fig.add_subplot(gs[1, 0:2]); letter(c, "C", x=-.09); bands(c)
    for n, cl, lw, nm in (("o", INK, 1.4, "data"), ("q0", GREY, 1.2, "independent model"), ("q1", PAIR, 1.2, "pairwise model")):
        M = np.array([np.mean([tc[f"{k}__pause__{n}"] for k in keys if k.startswith(s)], 0) for s in ses]); c.plot(off, np.median(M, 0), color=cl, lw=lw, label=nm)
    c.set_xlabel("time from peak velocity (ms)"); c.set_ylabel("P(population pause)\nper 10-ms bin, held out"); c.set_ylim(.09, .27)
    c.set_title("Population pauses (few cells active): data vs models (median over 21 sessions)")
    e = pd.read_csv(os.path.join(E15, "assets/tables/e15_conditions.csv")); e = e[e.test_window == "W"]
    d = fig.add_subplot(gs[:, 2]); letter(d, "D"); zero(d)
    for i, (ev, col_, cl, hol, nm) in enumerate((("pause", "L_obs0", GREY, False, "pauses vs\nindependent"), ("pause", "L_obs1", PAIR, False, "pauses vs\npairwise"),
                                                  ("burst", "L_obs0", GREY, True, "bursts vs\nindependent"), ("burst", "L_obs1", PAIR, True, "bursts vs\npairwise"))):
        dots(d, i, e[e.event == ev].groupby("session")[col_].mean(), cl, hollow=hol, seed=i)
    d.set_xticks(range(4), ["pause\nindep.", "pause\npairwise", "burst\nindep.", "burst\npairwise"], fontsize=6.3)
    d.set_ylabel("data / model at rest (log ratio)"); d.set_title("Excess joint events at rest")
    save(fig, "Fig5_fixed_couplings")


# ============================================================== Figure: stationary vs time-varying fields
def fig_stationary():
    A, keys = [], []
    for f in sorted(glob.glob(os.path.join(D, "regen/regen_*.npz"))):
        z = np.load(f); A.append(z["abs"]); keys += [int(str(k)[:6]) for k in z["keys"]]
    A = np.concatenate(A); keys = np.array(keys); t = (np.arange(700) - 400) * 10.0
    base = (t >= -1000) & (t <= -400); acc = (t >= -90) & (t <= 0); w = (t >= -1000) & (t <= 1500)
    dA = A - A[:, :, base].mean(2, keepdims=True)                       # change from baseline; channels f1, c1, fs, cs, fr
    ses = np.unique(keys); S = np.array([dA[keys == s].mean(0) for s in ses])
    fig = plt.figure(figsize=(7.2, 2.6)); gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, .9], wspace=.42)
    lim = None; axs = []
    for k, (fi, ci, title, L) in enumerate(((0, 1, "Time-varying fields", "A"), (2, 3, "Constant (stationary) fields", "B"))):
        a = fig.add_subplot(gs[0, k], sharey=axs[0] if axs else None); axs.append(a); letter(a, L); bands(a); zero(a)
        for arr, cl, lw in ((S[:, fi] + S[:, ci], INK, 1.2), (S[:, fi], FIELD, 1.4), (S[:, ci], COUP, 1.4)):
            a.plot(t[w], np.median(arr, 0)[w], color=cl, lw=lw)
        a.set_title(title); a.set_xlabel("time from peak velocity (ms)")
    axs[0].set_ylabel("change in energy per neuron\n(from baseline; median, 21 sessions)")
    c = fig.add_subplot(gs[0, 2]); letter(c, "C"); zero(c)
    vals = [S[:, ch][:, acc].mean(1) for ch in (0, 1, 2, 3)]
    for i, (v, cl, hol) in enumerate(zip(vals, (FIELD, COUP, FIELD, COUP), (False, False, True, True))):
        dots(c, i, v, cl, hollow=hol, seed=i)
    c.set_xticks(range(4), ["field", "coupling", "field", "coupling"], fontsize=6.5)
    c.text(.5, -.2, "time-varying", transform=c.get_xaxis_transform(), ha="center", fontsize=7, color=INK2)
    c.text(2.5, -.2, "stationary", transform=c.get_xaxis_transform(), ha="center", fontsize=7, color=INK2)
    c.set_ylabel("change during acceleration\n(per session)"); c.set_title("Where the drop is attributed")
    save(fig, "FigX_stationary_vs_timevarying")
    out = {n: (float(np.median(v)), int((v < 0).sum())) for n, v in zip(("tv_field", "tv_coupling", "st_field", "st_coupling"), vals)}
    print("acceleration change (median, sessions<0):", out)

# ============================================================== Supplementary
def figS1():
    """Small multiples: rows = shuffle type, columns = quantity; each panel = data (black) vs that shuffle (grey)."""
    files = sorted(glob.glob(os.path.join(D, "shuffle/eshuf_*.npz"))); t = (np.arange(700) - 400) * 10.0; w = (t >= -1000) & (t <= 1500)
    TS = [np.nanmean(np.load(f)["A"], 0) for f in files]
    cols = [(0, "field term", FIELD), (1, "coupling term", COUP), (2, "total energy", INK), (3, "firing rate", RATE)]
    rows = [(1, "reaches shuffled\nacross cells"), (2, "each cell shifted\nin time"), (3, "whole population\nshifted in time")]
    fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.0), sharex=True)
    for r, (ti, rlab) in enumerate(rows):
        for c, (ch, clab, cl) in enumerate(cols):
            a = ax[r, c]; bands(a)
            real = np.median([A[0, 0, ch] for A in TS], 0); shf = np.median([np.nanmean(A[ti, :, ch], 0) for A in TS], 0)
            a.plot(t[w], shf[w], color=GREY, lw=1.3); a.plot(t[w], real[w], color=cl, lw=1.3)
            if r == 0: a.set_title(clab); letter(a, "ABCD"[c], x=-.32)
            if c == 0: a.set_ylabel(rlab, fontsize=7)
            if r == 2: a.set_xlabel("time from peak\nvelocity (ms)")
    fig.tight_layout(); save(fig, "FigS1_shuffle_controls")

def figS2():
    r = pd.read_csv(os.path.join(E14, "assets/tables/e14_rows.csv")); n1 = r[(r.kind == "N1") & (r.stim == 0) & (r.window == "E")]
    fig, a = plt.subplots(figsize=(2.6, 2.4)); zero(a)
    dots(a, 0, n1.D_supp, INK, seed=0); dots(a, 1, n1.D_other, GREY, seed=1)
    a.set_xticks([0, 1], ["suppressed\npairs", "other\npairs"]); a.set_ylabel("change in 1-ms synchrony index\n(100 ms before endpoint − rest)")
    a.set_title("Millisecond synchrony (21 sessions)"); save(fig, "FigS2_ms_synchrony")

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2, figS3, fig_stationary): f()
