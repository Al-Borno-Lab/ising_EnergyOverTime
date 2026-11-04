"""Regenerate every exploratory/supporting figure of the evidence package (PNG + vector PDF) and their summary tables.
    python make_figures.py DATA_DIR REPO_DIR OUT_DIR
DATA_DIR = local copy of cluster outputs (entropy/, shuffle/, reachlock/, realign/, onems/, regen/, spikes_v2.npz)
REPO_DIR = repository root (for phase_landmarks.csv, h0_results.csv, E12 tables)
All analyses are EXPLORATORY unless stated; captions say so."""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D, REPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
STATS = {}

def save(fig, folder, name, note=None):
    d = os.path.join(OUT, folder); os.makedirs(d, exist_ok=True)
    if note: fig.text(0.01, -0.03, note, fontsize=7.5, color="#555", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def table(df, folder, name):
    d = os.path.join(OUT, folder); os.makedirs(d, exist_ok=True); df.to_csv(os.path.join(d, f"{name}.csv"), index=False)

def wil(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x != 0) else float("nan")

t700 = (np.arange(700) - 400) * 10

# ---------------------------------------------------------------- 2: P(K) difference view (data only, no fit)
def fig_pk():
    Z = np.load(os.path.join(D, "spikes_v2.npz")); key = "210614_0"
    X = Z["X_" + key][Z["exclude_" + key] == 0].reshape(-1, Z["X_" + key].shape[2]).astype(int)
    N = X.shape[1]; K = np.arange(N + 1)
    pk = lambda M: np.bincount(M.sum(1), minlength=N + 1) / len(M)
    real = pk(X); ind = np.array([1.0])
    for pi in X.mean(0): ind = np.convolve(ind, [1 - pi, pi])
    rng = np.random.default_rng(0)
    sh = np.array([pk(np.column_stack([rng.permutation(X[:, i]) for i in range(N)])) for _ in range(200)])
    lo, hi = np.percentile(sh - ind, [2.5, 97.5], axis=0); d = real - ind
    out = int(((d < lo) | (d > hi)).sum()); STATS["pk_K_outside_band"] = f"{out}/{N + 1}"
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    ax[0].plot(K, ind, color="#888", lw=2, label="independent cells, same rates"); ax[0].plot(K, real, color="#c0392b", lw=2, label="observed")
    ax[0].set_xlabel("K = cells active in the same 10-ms bin"); ax[0].set_ylabel("P(K)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("a  Linear scale", loc="left")
    ax[1].semilogy(K, ind, color="#888", lw=2); ax[1].semilogy(K, real, color="#c0392b", lw=2); ax[1].set_ylim(1e-6, .5)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("P(K), log scale"); ax[1].set_title("b  Log scale: the tails separate", loc="left")
    ax[2].axhline(0, color="k", lw=.7); ax[2].fill_between(K, lo, hi, color="#888", alpha=.3, label="95% range if independent (200 shuffles)")
    ax[2].bar(K, d, color=np.where(d > 0, "#c0392b", "#2f6db3"), width=.7, label="observed − independent")
    ax[2].set_xlabel("K"); ax[2].set_ylabel("difference in P(K)"); ax[2].legend(frameon=False, fontsize=7.5, loc="lower center")
    ax[2].set_title(f"c  Excess of all-together / all-quiet bins ({out}/{N + 1} K outside band)", loc="left")
    save(fig, "2_coupling_beyond_rates_at_rest", "P_K_difference_210614",
         f"Session 210614 baseline block, N = {N} cells, {len(X):,} bins (exclusions applied). Independent = Poisson-binomial with each cell's rate. "
         "Shuffle band: each cell's bins permuted independently. Note: this shuffle also removes reach-locked rate modulation.")

# ---------------------------------------------------------------- 1: synthetic validation (redrawn)
def fig_h0():
    d = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/h0_results.csv"))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sty = {(11, 20): ("#9ecae1", "o"), (11, 60): ("#2f6db3", "s"), (15, 20): ("#fdae6b", "^"), (15, 60): ("#e6550d", "D")}
    for (N, R), (c, m) in sty.items():
        s = d[(d.N == N) & (d.R == R)]; r = s[s.scen != "C"]; q = s[s.scen == "C"]
        ax.scatter(r.true_dI, r.est_dI, color=c, marker=m, s=30, label=f"N = {N} cells, {R} reaches")
        ax.scatter(q.true_dI, q.est_dI, facecolor="white", edgecolor=c, marker=m, s=55, lw=1.6, zorder=3)
    ax.plot([-.0105, .0035], [-.0105, .0035], "--", color="#777", lw=1, label="perfect agreement")
    ax.axhline(0, color="#bbb", lw=.6); ax.axvline(0, color="#bbb", lw=.6); ax.set_ylim(-.0105, .0045)
    for sc, g in d.groupby("scen"):
        STATS[f"h0_{sc}_sign_correct"] = f"{int((np.sign(g.true_dI) == np.sign(g.est_dI)).sum())}/{len(g)}"
        STATS[f"h0_{sc}_within50pct"] = f"{int(((g.est_dI - g.true_dI).abs() <= .5 * g.true_dI.abs()).sum())}/{len(g)}"
        STATS[f"h0_{sc}_median_abs_true_dI"] = float(g.true_dI.abs().median())
    ax.text(.0105, -.0072, "Hollow = scenario C: couplings themselves changed.\nThe fixed-coupling measure is not built to see that\nand reports ~0 instead.", fontsize=8.5, color="#444", va="top")
    ax.set_xlabel("true change in multi-information, reach vs rest (bits per cell)"); ax.set_ylabel("estimated change, fixed-coupling model (bits per cell)")
    ax.set_title("Rate-driven scenarios (filled) fall on or near the dashed line", loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "1_model_and_validation", "synthetic_validation_key_measure", "Method validation on SIMULATED data (E09 H0). Source: 09_claude_science_handoff/assets/tables/h0_results.csv.")

# ---------------------------------------------------------------- regen-based: absolute energy, overlay, field-dip shuffles, burster/pauser
def load_regen():
    A, O, keys = [], [], []
    for f in sorted(glob.glob(os.path.join(D, "regen/regen_*.npz"))):
        z = np.load(f); A.append(z["abs"]); O.append(z["ovl"]); keys += list(z["keys"])
    return np.concatenate(A), np.concatenate(O), keys

def fig_energy_abs(A, keys):
    ses = sorted(set(k.split("_")[0] for k in keys)); idx = {s: [i for i, k in enumerate(keys) if k.startswith(s)] for s in ses}
    agg = {n: np.array([A[idx[s], j].mean(0) for s in ses]) for j, n in enumerate(("f1", "c1", "fs", "cs", "fr"))}
    agg["t1"] = agg["f1"] + agg["c1"]; agg["ts"] = agg["fs"] + agg["cs"]
    win = (t700 >= -1000) & (t700 <= 2000); base = (t700 >= -1000) & (t700 <= -400)
    for n in agg:
        m = np.median(agg[n], 0); STATS[f"energy_{n}_baseline"] = round(float(m[base].mean()), 3)
        STATS[f"energy_{n}_min_ms"] = int(t700[win][m[win].argmin()]); STATS[f"energy_{n}_min"] = round(float(m[win].min()), 3)
    table(pd.DataFrame({"t_ms": t700[win], **{f"{n}_median": np.median(agg[n], 0)[win] for n in agg}}), "3_rates_vs_coupling_over_reach", "energy_absolute_medians")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for a, (ttl, ff, cc, tt) in zip(ax, [("Time-varying-field model (M1)", "f1", "c1", "t1"), ("Stationary model (Ms)", "fs", "cs", "ts")]):
        for n, lab, col in [(tt, "total energy", "#222"), (ff, "field term −Σh·s", "#e0861f"), (cc, "coupling term −ΣJ·s·s", "#7b3294")]:
            m = np.median(agg[n], 0)[win]; lo, hi = np.percentile(agg[n], [25, 75], 0)
            a.plot(t700[win], m, color=col, lw=2, label=lab); a.fill_between(t700[win], lo[win], hi[win], color=col, alpha=.12, lw=0)
        a.axvspan(-90, 0, color="#cfdcef", alpha=.6, lw=0); a.axvspan(0, 120, color="#f2d3d3", alpha=.6, lw=0)
        a.axvline(0, color="k", lw=.7, ls=":"); a.set_xlabel("time from peak velocity (ms)"); a.set_title(ttl, loc="left")
    ax[0].set_ylabel("energy per neuron (median over 21 sessions, IQR)")
    a2 = ax[0].twinx(); a2.spines["right"].set_visible(True)
    m = np.median(agg["fr"], 0)[win]; lo, hi = np.percentile(agg["fr"], [25, 75], 0)
    a2.plot(t700[win], m, color="#c0392b", lw=1.6, label="population firing rate (right axis)"); a2.fill_between(t700[win], lo[win], hi[win], color="#c0392b", alpha=.08, lw=0)
    a2.set_ylabel("spike probability per 10-ms bin", color="#c0392b"); a2.tick_params(axis="y", colors="#c0392b")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
    save(fig, "3_rates_vs_coupling_over_reach", "energy_absolute_with_firing_rate",
         "Absolute energy (not baseline-subtracted); lower = more probable pattern. Pseudo-likelihood fits, exclusions applied; blocks averaged within session. Blue/red = acceleration/deceleration.")

def fig_overlay(O):
    t = np.arange(-60, 101) * 10; sty = {0: ("Field term, Ising (coupled)", "#1f4e99", "-"), 1: ("Field term, no coupling", "#e0861f", "--"),
                                          2: ("Polarization index (no model)", "#2a9d8f", "-"), 3: ("Average firing rate", "#c0392b", "-")}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    for j, (lab, c, ls) in sty.items():
        M = O[:, j]; m, se = M.mean(0), M.std(0) / np.sqrt(len(M))
        ax[0].plot(t, m, color=c, ls=ls, lw=2, label=lab); ax[0].fill_between(t, m - se, m + se, color=c, alpha=.15, lw=0)
    ax[0].axvline(0, color="k", ls=":", lw=.8); ax[0].axvline(40, color="#888", ls=":", lw=.8)
    ax[0].set_xlabel("time from peak acceleration (ms)"); ax[0].set_ylabel("change from baseline (baseline z)"); ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_title("a  Field term tracks polarization; mean rate does not", loc="left")
    per = {j: np.array([np.corrcoef(O[i, 0], O[i, j])[0, 1] for i in range(len(O))]) for j in (1, 2, 3)}
    for n, j in zip(("FT0", "POLb", "FR"), (1, 2, 3)): STATS[f"overlay_median_r_FT_vs_{n}"] = round(float(np.median(per[j])), 3)
    for x, j in enumerate((1, 2, 3)):
        v = per[j]; ax[1].scatter(x + np.random.default_rng(j).uniform(-.12, .12, len(v)), v, s=10, color=sty[j][1], alpha=.6)
        ax[1].plot([x - .25, x + .25], [np.median(v)] * 2, color="k", lw=2); ax[1].text(x + .28, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=8)
    ax[1].set_xticks([0, 1, 2], ["Field term,\nno coupling", "Polarization\nindex", "Average\nfiring rate"]); ax[1].axhline(0, color="#999", lw=.6); ax[1].set_ylim(-1.05, 1.05)
    ax[1].set_ylabel("correlation with Ising field-term trace\n(per condition, −600 to +1000 ms)"); ax[1].set_title("b  Similarity to the field term", loc="left")
    save(fig, "4_polarization_at_peak_acceleration", "field_term_vs_polarization_vs_rate",
         "63 conditions, exclusions applied; baseline-z traces aligned to each condition's peak acceleration (mean ± s.e.m.). EXPLORATORY (E12 pipeline).")

def fig_fielddip_shuffles():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "regen/fielddip_shuffles_*.csv"))]); d["session"] = d.key.str.split("_").str[0]
    s = d.groupby(["kind", "session"])[["dist", "hit"]].mean().reset_index(); real = s[s.kind == "real"].set_index("session")
    rows = []
    for k in ("real", "trial_shuffle", "neuron_time_shift", "label_shuffle"):
        x = s[s.kind == k].set_index("session").loc[real.index]
        rows.append(dict(kind=k, median_dist_ms=x.dist.median(), hit_rate=x.hit.mean(),
                         p_dist_vs_real=np.nan if k == "real" else wil(real.dist - x.dist), p_hit_vs_real=np.nan if k == "real" else wil(real.hit - x.hit)))
    T = pd.DataFrame(rows); table(T, "4_polarization_at_peak_acceleration", "fielddip_shuffle_controls"); STATS["fielddip_shuffles"] = T.round(4).to_dict("records")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6)); lab = ["Real", "Reaches\nshuffled\nacross cells", "Each cell\nshifted in time\n(±600 ms)", "Another\nsession's\nlandmarks"]
    for j, (col, yl) in enumerate([("hit", "fraction of dips in the 200 ms\nbefore peak velocity"), ("dist", "distance of dip from\npeak acceleration (ms)")]):
        for i, k in enumerate(("real", "trial_shuffle", "neuron_time_shift", "label_shuffle")):
            v = s[s.kind == k][col].values; ax[j].scatter(i + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=14, color=["#1f4e99", "#888", "#c0392b", "#bbb"][i], alpha=.8)
            ax[j].plot([i - .25, i + .25], [v.mean() if col == "hit" else np.median(v)] * 2, color="k", lw=2)
        ax[j].set_xticks(range(4), lab, fontsize=7.5); ax[j].set_ylabel(yl)
    ax[0].set_title("a  Dip lands before peak velocity", loc="left"); ax[1].set_title("b  Distance from peak acceleration", loc="left")
    save(fig, "4_polarization_at_peak_acceleration", "fielddip_shuffle_controls",
         "EXPLORATORY. Field-term dip timing (E12 search) for real data and 5 shuffles of each type per block; one dot per session. "
         "Landmark shuffle is weak because peak acceleration sits at nearly the same bin in every session.")

def fr_minmax():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "regen/fr_minmax_*.csv"))])
    d["d_match"] = np.where(d.type == "pauser", d.d_min, d.d_max); d["d_other"] = np.where(d.type == "pauser", d.d_max, d.d_min)
    s = d.groupby("session")[["d_min", "d_max", "d_match", "d_other"]].median()
    T = pd.DataFrame([dict(n_conditions=len(d), bursters=int((d.type == "burster").sum()), pausers=int((d.type == "pauser").sum()),
                           median_matched_ms=s.d_match.median(), median_other_ms=s.d_other.median(), p_matched_vs_other=wil(s.d_match - s.d_other),
                           ft_trough_median_ms=d.ft_t.median())])
    table(d, "4_polarization_at_peak_acceleration", "burster_pauser_by_condition"); table(T, "4_polarization_at_peak_acceleration", "burster_pauser_summary")
    STATS["burster_pauser"] = T.round(4).to_dict("records")[0]

# ---------------------------------------------------------------- S2: energy time-series shuffle controls
def fig_shuffle_timeseries():
    files = sorted(glob.glob(os.path.join(D, "shuffle/eshuf_*.npz")))
    base = (t700 >= -1000) & (t700 <= -400); rw = (t700 >= -300) & (t700 <= 100); full = (t700 >= -1000) & (t700 <= 2000)
    FEAT = {"field_dip": (0, "dip"), "total_dip": (2, "dip"), "fr_change": (3, "absdip"), "coupling_level": (1, "level")}
    rows, TS = [], {}
    for f in files:
        z = np.load(f); A = np.nanmean(z["A"], 0); types = list(z["types"]); s = os.path.basename(f)[6:12]; TS[s] = A
        for name, (ch, kind) in FEAT.items():
            x = A[:, :, ch, :]
            v = (x[..., rw].mean(-1) - x[..., base].mean(-1)) if kind == "dip" else np.abs(x[..., rw].mean(-1) - x[..., base].mean(-1)) if kind == "absdip" else x[..., full].mean(-1)
            real = v[0, 0]
            for ti, ty in enumerate(types[1:], 1):
                sh = v[ti][np.isfinite(v[ti])]
                beyond = (abs(real) > np.abs(sh).max()) if kind != "level" else (real < sh.min())
                rows.append(dict(session=s, feature=name, shuffle=ty, real=real, shuf_median=np.median(sh), beyond_all_20=bool(beyond)))
    R = pd.DataFrame(rows); table(R, "S2_shuffle_controls_energy_timeseries", "shuffle_test_by_session")
    S = pd.DataFrame([dict(feature=f, shuffle=t, n=len(g), sessions_beyond_all_20=int(g.beyond_all_20.sum()), median_real=g.real.median(),
                           median_shuffle=g.shuf_median.median(), p_wilcoxon=wil(g.real - g.shuf_median)) for (f, t), g in R.groupby(["feature", "shuffle"])])
    table(S, "S2_shuffle_controls_energy_timeseries", "shuffle_test_summary"); STATS["energy_shuffles"] = S.round(4).to_dict("records")
    lab = ["field term (−Σh·s)", "coupling term (−ΣJ·s·s)", "total energy", "population firing rate"]
    col = {"real": "#111", "trial": "#7b3294", "neuronshift": "#e0861f", "popshift": "#2a9d8f"}
    nam = {"real": "real data", "trial": "trial shuffle", "neuronshift": "per-neuron time shift", "popshift": "whole-population time shift"}
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    for ch in range(4):
        for ti, ty in enumerate(["real", "trial", "neuronshift", "popshift"]):
            M = np.array([TS[s][0, 0, ch] if ty == "real" else np.nanmean(TS[s][ti, :, ch], 0) for s in TS])
            ax[ch].plot(t700[full], np.median(M, 0)[full], color=col[ty], lw=2 if ty == "real" else 1.4, label=nam[ty])
        ax[ch].axvline(0, color="k", ls=":", lw=.7); ax[ch].axvspan(-300, 100, color="#eee", lw=0); ax[ch].set_title(lab[ch], loc="left"); ax[ch].set_xlabel("time from peak velocity (ms)")
    ax[0].set_ylabel("per neuron (median over 21 sessions)"); ax[3].set_ylabel("spike probability per 10-ms bin"); ax[0].legend(frameon=False, fontsize=8)
    save(fig, "S2_shuffle_controls_energy_timeseries", "energy_timeseries_vs_shuffles",
         "EXPLORATORY. M1 refit on each shuffle (20 per type per block); shuffle curves = mean over shuffles, median over sessions. In the FR panel the real curve lies under the trial-shuffle curve (identical by construction).")

# ---------------------------------------------------------------- S3: 1 ms
def fig_1ms():
    f1, c1, r1, f10, c10, r10 = ([] for _ in range(6))
    for f in sorted(glob.glob(os.path.join(D, "onems/res_*.npz"))):
        z = np.load(f, allow_pickle=True); keys = sorted(set(k.split("__")[0] for k in z.files if "__" in k))
        f1.append(np.mean([z[f"{k}__f1"] for k in keys], 0)); c1.append(np.mean([z[f"{k}__c1"] for k in keys], 0)); r1.append(np.mean([z[f"{k}__fr1"] for k in keys], 0))
        f10.append(np.mean([z[f"{k}__f10"] for k in keys], 0)); c10.append(np.mean([z[f"{k}__c10"] for k in keys], 0)); r10.append(np.mean([z[f"{k}__fr10"] for k in keys], 0))
    f1, c1, r1, f10, c10, r10 = map(np.array, (f1, c1, r1, f10, c10, r10)); STATS["onems_sessions"] = len(f1)
    t1 = (np.arange(7000) - 4000) * 1.0
    def scaled(M, t, b0, b1):
        m = np.median(M, 0); b = m[(t >= b0) & (t <= b1)].mean(); d = m - b; return d / np.max(np.abs(d[(t >= -400) & (t <= 800)]))
    sm1 = lambda a: np.convolve(a, np.ones(10) / 10, mode="same")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (M1, M10, lab) in zip(ax, [(f1, f10, "field term"), (c1, c10, "coupling term"), (r1, r10, "population firing rate")]):
        w10 = (t700 >= -400) & (t700 <= 800); w1 = (t1 >= -400) & (t1 <= 800)
        a.plot(t700[w10], scaled(M10, t700, -1000, -400)[w10], color="#333", lw=2, label="10-ms fit")
        a.plot(t1[w1], sm1(scaled(M1, t1, -1000, -400))[w1], color="#2f6db3", lw=1.2, label="1-ms fit (10-ms running mean)")
        a.axvline(0, color="k", ls=":", lw=.7); a.axvline(-40, color="#2f6db3", ls="--", lw=.8); a.axhline(0, color="#aaa", lw=.6)
        a.set_title(f"{lab}: change from baseline, scaled", loc="left"); a.set_xlabel("time from peak velocity (ms)")
        STATS[f"onems_{lab.split()[0]}_min_ms_1ms"] = int(t1[w1][sm1(scaled(M1, t1, -1000, -400))[w1].argmin()])
        STATS[f"onems_{lab.split()[0]}_min_ms_10ms"] = int(t700[w10][scaled(M10, t700, -1000, -400)[w10].argmin()])
    ax[0].legend(frameon=False, fontsize=8)
    save(fig, "S3_bin_size_1ms", "compare_1ms_vs_10ms_scaled",
         "EXPLORATORY. Same model fitted at 1-ms and 10-ms bins (21 sessions; 1-ms spikes re-binned reproduce the 10-ms data exactly). Curves = change from baseline "
         "(−1000..−400 ms), median over sessions, scaled to each curve's max |change|; energy scales are not comparable across bin sizes. Dashed = median peak acceleration.")
    w = (t1 >= -1000) & (t1 <= 2000)
    fig, ax = plt.subplots(figsize=(8, 4))
    for M, lab, col in [(f1 + c1, "total energy", "#222"), (f1, "field term", "#e0861f"), (c1, "coupling term", "#7b3294")]:
        ax.plot(t1[w], sm1(np.median(M, 0))[w], color=col, lw=1.4, label=lab)
    a2 = ax.twinx(); a2.spines["right"].set_visible(True); a2.plot(t1[w], sm1(np.median(r1, 0))[w], color="#c0392b", lw=1, label="firing rate (right)")
    a2.set_ylabel("spike probability per 1-ms bin", color="#c0392b"); ax.set_ylabel("energy per neuron per 1-ms bin"); ax.set_xlabel("time from peak velocity (ms)")
    ax.axvline(0, color="k", ls=":", lw=.7); ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "S3_bin_size_1ms", "energy_absolute_1ms", "EXPLORATORY. 1-ms version of the absolute-energy panel (10-ms running mean for display); median over 21 sessions.")
    gates = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "onems/gate_*.json")))]
    table(pd.DataFrame(gates), "S3_bin_size_1ms", "gate_report")

# ---------------------------------------------------------------- S4: entropy + baseline reference
def fig_entropy():
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "entropy/entropy_timecourse_*.csv")))])
    E["I_per_N"] = E.Sind_per_N - E.S_per_N; win = (E.t_ms >= -1000) & (E.t_ms <= 2000)
    COL = {0: "#2f6db3", 1: "#c0392b", 2: "#2a9d8f"}; NAME = {0: "stim 0 (baseline)", 1: "stim 1 (stimulation)", 2: "stim 2 (washout)"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (c, lab) in enumerate([("S_per_N", "entropy per neuron (bits / 10-ms bin)"), ("Sind_per_N", "independent entropy per neuron (bits)"), ("I_per_N", "multi-information per neuron (bits)")]):
        for st in (0, 1, 2):
            g = E[win & (E.stim == st)].groupby(["session", "t_ms"])[c].mean().unstack(); m = g.median(0)
            ax[j].plot(m.index, m.values, color=COL[st], lw=1.8, label=NAME[st]); ax[j].fill_between(m.index, g.quantile(.25, 0), g.quantile(.75, 0), color=COL[st], alpha=.1, lw=0)
        ax[j].axvline(0, color="k", ls=":", lw=.7); ax[j].set_xlabel("time from peak velocity (ms)"); ax[j].set_ylabel(lab)
    ax[0].legend(frameon=False, fontsize=8)
    w = E[win].groupby(["session", "stim"]).agg(I=("I_per_N", "mean"), S=("S_per_N", "mean")).unstack()
    STATS["entropy_block_tests"] = {f"{m}_stim{a}-stim{b}": round(wil(w[m][a] - w[m][b]), 3) for a, b in ((1, 0), (2, 0), (2, 1)) for m in ("I", "S")}
    save(fig, "S4_entropy_and_learning_exploratory", "entropy_by_block",
         "EXPLORATORY. Each block fitted separately; entropy exact from the fitted model (sessions > 16 cells: one fixed 16-cell subset). Median over 21 sessions, IQR.")
    B = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "entropy/baseline_ref_surprise_*.csv")))])
    for m in ("M1", "M0"):
        b = B[B.stim == 0].groupby("session")[f"sur_{m}_reach"].mean(); B[f"rel_{m}"] = B[f"sur_{m}_reach"] - B.session.map(b)
    B["rel_cpl"] = B.rel_M1 - B.rel_M0
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True); x0 = 0; offs = {}
    for st in (0, 1, 2):
        nmax = int(B[B.stim == st].groupby("session").reach_idx.max().median()) + 1; offs[st] = x0
        for a, c, lab in [(ax[0], "rel_M1", "surprise under the baseline model\n(bits/neuron/bin, minus held-out baseline mean)"), (ax[1], "rel_cpl", "part due to coupling\n(full − rates-only change)")]:
            g = B[B.stim == st].pivot_table(index="reach_idx", columns="session", values=c); g = g[g.notna().sum(1) >= 10]
            a.plot(x0 + g.index.values, g.median(1).values, "o-", color=COL[st], ms=3.5, lw=1.4, label=NAME[st]); a.fill_between(x0 + g.index.values, g.quantile(.25, 1), g.quantile(.75, 1), color=COL[st], alpha=.15, lw=0); a.set_ylabel(lab)
        x0 += nmax + 2
    for a in ax:
        a.axhline(0, color="#999", lw=.6)
        for st in (1, 2): a.axvline(offs[st] - 1, color="#bbb", ls="--", lw=1)
    ax[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); ax[1].set_xlabel("reach number within session (baseline → stimulation → washout)")
    save(fig, "S4_entropy_and_learning_exploratory", "baseline_referenced_surprise",
         "EXPLORATORY. One model per session fitted on baseline reaches; later reaches scored against it (exact likelihood, ±200 ms around the crossing); baseline scored held-out. "
         "CAVEAT: confounded with time in session (recording drift / animal state).")
    ses = B.groupby(["session", "stim"]).rel_M1.mean().unstack(); rows = []
    for st in (1, 2): rows.append(dict(test=f"stim{st} vs baseline", n=int(ses[st].notna().sum()), median=ses[st].median(), n_pos=int((ses[st] > 0).sum()), p=wil(ses[st])))
    for st in (0, 1, 2):
        r = np.array([spearmanr(d.reach_idx, d.rel_M1).correlation for _, d in B[B.stim == st].groupby("session") if len(d) >= 6])
        rows.append(dict(test=f"trend within stim{st}", n=len(r), median=np.nanmedian(r), n_pos=int((r > 0).sum()), p=wil(r)))
    cp = B.groupby(["session", "stim"]).rel_cpl.mean().unstack()
    for st in (1, 2): rows.append(dict(test=f"coupling part stim{st}", n=int(cp[st].notna().sum()), median=cp[st].median(), n_pos=int((cp[st] > 0).sum()), p=wil(cp[st])))
    T = pd.DataFrame(rows); table(T, "S4_entropy_and_learning_exploratory", "baseline_ref_stats"); STATS["baseline_ref"] = T.round(4).to_dict("records")

# ---------------------------------------------------------------- S5: per-reach locking + re-alignment
def fig_reach():
    R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(D, "reachlock/reach_lock_by_condition_*.csv"))]); rows = []
    for sig, g in R.groupby("signal"):
        s = g.groupby("session").agg(rho=("rho", "mean"), rhon=("rho_null_mean", "mean"), d=("dist_ms", "mean"), dn=("dist_null_mean", "mean"))
        rows.append(dict(signal=sig, rho_median=s.rho.median(), rho_null=s.rhon.median(), rho_gt_null=int((s.rho > s.rhon).sum()), p_rho=wil(s.rho - s.rhon),
                         dist_ms=s.d.median(), dist_null_ms=s.dn.median(), closer=int((s.d < s.dn).sum()), p_dist=wil(s.d - s.dn), blocks_p_rho_lt05=int((g.p_rho < .05).sum())))
    T = pd.DataFrame(rows); table(T, "S5_per_reach_locking_exploratory", "per_reach_correlation_summary"); STATS["reach_lock"] = T.round(4).to_dict("records")
    RA = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "realign/realign_[0-9]*.csv")))]); curves = {}
    for f in sorted(glob.glob(os.path.join(D, "realign/realign_curves_*.npz"))):
        z = np.load(f)
        for k in z["keys"]: curves[str(k)] = {n: z[f"{n}__{k}"] for n in ("field_own", "field_null", "total_own", "total_null")}
    rows = []
    for sig in ("field", "total"):
        own = RA.groupby("session")[f"{sig}_depth_own"].mean(); nul = RA.groupby("session")[f"{sig}_depth_null_mean"].mean()
        rows.append(dict(signal=sig, depth_own=own.median(), depth_shuffled=nul.median(), own_deeper=int((own < nul).sum()), p=wil(own - nul)))
    T2 = pd.DataFrame(rows); table(T2, "S5_per_reach_locking_exploratory", "realign_summary"); STATS["realign"] = T2.round(4).to_dict("records")
    STATS["peak_acc_jitter_sd_ms"] = round(float(RA.pacc_sd_ms.median()), 1)
    w = (t700 >= -600) & (t700 <= 600); fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, sig in zip(ax, ("field", "total")):
        for kind, col, nm in (("own", "#111", "aligned to OWN peak acceleration"), ("null", "#c0392b", "aligned to SHUFFLED reach's peak acceleration")):
            M = np.array([np.mean([curves[k][f"{sig}_{kind}"] for k in g.key], 0) for _, g in RA.groupby("session")]); M = M - M[:, 300:360].mean(1, keepdims=True)
            a.plot(t700[w], np.median(M, 0)[w], color=col, lw=2 if kind == "own" else 1.5, label=nm)
        a.axvline(0, color="k", ls=":", lw=.7); a.set_xlabel("time from peak acceleration (ms)"); a.set_ylabel(f"change in {sig} (per neuron)"); a.set_title(f"{'ab'[sig == 'total']}  {sig}", loc="left")
    ax[0].legend(frameon=False, fontsize=8)
    save(fig, "S5_per_reach_locking_exploratory", "realign_own_vs_shuffled",
         "EXPLORATORY. Per-reach energy aligned to each reach's own (black) or a shuffled reach's (red; 1,000 shuffles) peak acceleration, averaged. Low power: peak-acceleration jitter ~±55 ms vs a ~300-ms-wide dip.")

if __name__ == "__main__":
    fig_pk(); fig_h0()
    if glob.glob(os.path.join(D, "regen/regen_*.npz")):
        A, O, keys = load_regen(); fig_energy_abs(A, keys); fig_overlay(O); fig_fielddip_shuffles(); fr_minmax()
    fig_shuffle_timeseries(); fig_1ms(); fig_entropy(); fig_reach()
    json.dump(STATS, open(os.path.join(OUT, "regenerated_stats.json"), "w"), indent=1, default=str); print(json.dumps(STATS, indent=1, default=str)[:4000])
