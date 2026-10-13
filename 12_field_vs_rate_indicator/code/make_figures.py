"""E12 figures from results/primary (traces.npz, session_metrics.csv, primary_family.csv)."""
import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--assets", required=True)
a = ap.parse_args(); P = f"{a.results}/primary"
tr = np.load(f"{P}/traces.npz"); sm = pd.read_csv(f"{P}/session_metrics.csv"); fam = pd.read_csv(f"{P}/primary_family.csv")
lm = pd.read_csv("../../09_claude_science_handoff/assets/tables/phase_landmarks.csv"); lm["cond"] = lm.session.astype(str) + "_" + lm.stim.astype(str)
sigs = ["FT", "FTu", "FT0", "POLb", "dFR", "FR", "TRAJ", "DEC"]
col = dict(FT="#1b4f9c", FTu="#5b8fd6", FT0="#e08a1e", POLb="#2a9d8f", dFR="#8a5aa8", FR="#c0392b", TRAJ="#777777", DEC="#222222")
rel = np.arange(-60, 101)

# Fig 1: mean z-scored trace aligned to peak acceleration
fig, ax = plt.subplots(2, 4, figsize=(14, 5.5), sharex=True)
for k, s in enumerate(sigs):
    A = ax.flat[k]; M = []
    for _, r in lm.iterrows():
        key = f"{r.cond}__{s}"
        if key in tr: M.append(tr[key][r.peak_acc + rel] if s != "DEC" else tr[key][r.peak_acc + rel])
    M = np.array(M); z = M if s == "DEC" else M
    A.plot(rel * 10, z.mean(0), color=col[s]); A.fill_between(rel * 10, z.mean(0) - z.std(0) / np.sqrt(len(z)), z.mean(0) + z.std(0) / np.sqrt(len(z)), color=col[s], alpha=.25)
    A.axvline(0, color="k", lw=.7, ls="--"); A.set_title(s + (" (decoder score)" if s == "DEC" else " (baseline z)")); A.set_xlabel("ms from peak acceleration")
fig.suptitle("E12: condition-mean signals around peak acceleration (63 conditions; mean ± s.e.m.)"); fig.tight_layout()
fig.savefig(f"{a.assets}/figures/signals_aligned.png", dpi=150); fig.savefig(f"{a.assets}/figures/signals_aligned.pdf")

# Fig 2: T1 and T2 per signal
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = ["FT", "FT0", "POLb", "dFR", "FR", "TRAJ", "DEC"]
for i, s in enumerate(order):
    d = sm[sm.signal == s]
    ax[0].bar(i - .18, d.hit.mean(), .35, color=col[s]); ax[0].bar(i + .18, d.hit_c.mean(), .35, color="lightgray")
    ax[0].plot(np.full(len(d), i - .18) + np.random.RandomState(i).uniform(-.06, .06, len(d)), d.hit, ".", color="k", ms=3)
    ax[1].boxplot(d.dev_ms, positions=[i], widths=.5, showfliers=False)
    ax[1].plot(np.full(len(d), i) + np.random.RandomState(i).uniform(-.1, .1, len(d)), d.dev_ms, ".", color=col[s], ms=4)
for A in ax: A.set_xticks(range(len(order))); A.set_xticklabels(order)
ax[0].set_ylabel("hit fraction ([-200,0] ms before peak vel.)", fontsize=9); ax[0].set_title("T1: real (colour) vs rest control (grey)", fontsize=10)
ax[1].set_ylabel("|extremum - peak acc.| (ms), session median", fontsize=9); ax[1].set_title("T2: timing precision (lower is better)")
fig.tight_layout(); fig.savefig(f"{a.assets}/figures/T1_T2_by_signal.png", dpi=150); fig.savefig(f"{a.assets}/figures/T1_T2_by_signal.pdf")

# Fig 3: forest of FT vs others
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for A, t, u, lab in ((ax[0], "T1", 1, "T1: FT minus other, hit fraction"), (ax[1], "T2", 1, "T2: FT minus other, ms (negative = FT more precise)")):
    d = fam[fam.test == t].reset_index(drop=True)
    for i, r in d.iterrows():
        A.errorbar(r.mean_diff, i, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt="o", color="k")
    A.set_yticks(range(len(d))); A.set_yticklabels(d.B); A.axvline(0, color="k", lw=.7); A.set_title(lab + " (95% bootstrap CI)")
    if t == "T1":
        A.axvspan(-.1, .1, color="gray", alpha=.15)
    else:
        A.axvspan(-15, 15, color="gray", alpha=.15)
fig.tight_layout(); fig.savefig(f"{a.assets}/figures/primary_contrasts_forest.png", dpi=150); fig.savefig(f"{a.assets}/figures/primary_contrasts_forest.pdf")
print("figures written")
