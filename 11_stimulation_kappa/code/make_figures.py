"""Figures for RESULTS.md from assets/tables/{session_table.csv,stats_summary.json}. usage: python make_figures.py <assets_dir> <results_dir>"""
import sys, os, csv, json, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
A, R = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(A, "tables", "session_table.csv"))))
S = json.load(open(os.path.join(A, "tables", "stats_summary.json")))
os.makedirs(os.path.join(A, "figures"), exist_ok=True)
col = {"2021": "#1b6ca8", "2022": "#c4561a"}
# Figure 1: per-session raw dkappa (primary, A4)
rows.sort(key=lambda r: float(r["dkappa_raw_primary"]))
fig, ax = plt.subplots(figsize=(6.4, 5.6))
y = np.arange(len(rows)); v = np.array([float(r["dkappa_raw_primary"]) for r in rows])
ax.axvspan(-0.3, 0.3, color="0.93", zorder=0); ax.axvline(0, color="0.4", lw=0.8)
ax.scatter(v, y, c=[col[r["cohort"]] for r in rows], zorder=3, s=36)
ax.set_yticks(y); ax.set_yticklabels(["%s %s" % (r["session"], r["mouse"]) for r in rows], fontsize=7)
s1 = S["S1"]; ax.axvline(s1["hl"], color="k", lw=1.4)
ax.axvspan(s1["ci90"][0], s1["ci90"][1], color="k", alpha=0.12)
ax.set_xlabel("raw Δκ = κ_pulse − κ_control (per session; primary)")
ax.set_title("S1: pulse minus control coupling scale, 21 sessions\nline = Hodges–Lehmann %.2f, band = 90%% CI [%.2f, %.2f], grey = ±0.3" % (s1["hl"], *s1["ci90"]), fontsize=9)
ax.scatter([], [], c=col["2021"], label="2021 cohort"); ax.scatter([], [], c=col["2022"], label="2022 cohort"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(A, "figures", "fig1_S1_dkappa_per_session.png"), dpi=150); plt.close()
# Figure 2: S1 and its sensitivity variants (HL with 90% CI)
keys = [("S1", "primary: raw (pseudo-onset control)"), ("diag_null_corrected", "diagnostic: null-corrected"), ("sens_round", "onset rounded"), ("sens_shift1", "onset floor + 1"),
        ("sens_fixedctrl", "fixed control 401–406"), ("sens_allreach", "all reaches"), ("sens_N5", "without 210622, 210623 (N<5)"),
        ("sens_without_seen_sessions", "without 210511, 220519"), ("cohort_2021", "2021 cohort only"), ("cohort_2022", "2022 cohort only")]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for i, (k, lab) in enumerate(keys[::-1]):
    t = S[k]; ax.errorbar(t["hl"], i, xerr=[[t["hl"] - t["ci90"][0]], [t["ci90"][1] - t["hl"]]], fmt="o", color="k", capsize=3)
    ax.text(1.0, i, ("n=%d p=%.3f" % (t["n"], t["p"])) if "p" in t else "n=%d (diagnostic)" % t["n"], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7)
ax.axvspan(-0.3, 0.3, color="0.93", zorder=0); ax.axvline(0, color="0.4", lw=0.8)
ax.set_yticks(range(len(keys))); ax.set_yticklabels([l for _, l in keys[::-1]], fontsize=8)
ax.set_xlabel("Hodges–Lehmann Δκ with 90% CI"); ax.set_title("S1 and sensitivity analyses (raw Δκ unless stated)", fontsize=9)
ax.set_xlim(-0.8, 0.9); plt.subplots_adjust(right=0.8); plt.tight_layout(rect=(0, 0, 0.88, 1)); plt.savefig(os.path.join(A, "figures", "fig2_S1_sensitivity.png"), dpi=150); plt.close()
# Figure 3: S2 per session (dI/N, dchi/N)
by = collections.defaultdict(list)
for f in glob.glob(os.path.join(R, "tasks", "*_primary.json")):
    r = json.load(open(f))
    if r["converged"] and all(map(np.isfinite, (r["dI_per_N"], r["dchiJ_per_N"], *r["kappa"]))): by[r["session"]].append((r["dI_per_N"], r["dchiJ_per_N"]))
ses = sorted(by); dI = [np.median([a for a, _ in by[s]]) for s in ses]; dc = [np.median([b for _, b in by[s]]) for s in ses]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, vals, name, k in ((axs[0], dI, "Δ multi-information / N (bits)", "S2_dI_per_N"), (axs[1], dc, "Δ χ_J / N", "S2_dchiJ_per_N")):
    ax.axvline(0, color="0.4", lw=0.8); ax.scatter(vals, range(len(ses)), c=[col["2021" if s.startswith("21") else "2022"] for s in ses], s=22)
    ax.axvline(S[k]["hl"], color="k"); ax.set_xlabel(name); ax.set_title("pulse − control window; HL %.4f, p=%.2f" % (S[k]["hl"], S[k]["p"]), fontsize=8)
axs[0].set_yticks(range(len(ses))); axs[0].set_yticklabels(ses, fontsize=6); axs[1].set_yticks([])
plt.tight_layout(); plt.savefig(os.path.join(A, "figures", "fig3_S2_thermo_per_session.png"), dpi=150); plt.close()
print("figures written")
