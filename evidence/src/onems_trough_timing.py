"""EXPLORATORY (run after seeing the 1-ms median curve). Timing of the field-term minimum at 1 ms, per condition and in the group median.
    python onems_trough_timing.py DATA_DIR REPO_DIR OUT_DIR
(1) Per condition (same test as the 10-ms timing result): minimum of the 5-ms-smoothed field term in -600..+1000 ms around peak velocity;
    hit = minimum within the 200 ms before peak velocity; rest = same search around -2000 and -2500 ms. Session-level Wilcoxon (reach - rest).
(2) Group median: session curves (change from baseline, scaled as in the six-panel figure), median over sessions, minimum in -200..+100 ms;
    leave-one-session-out range and 2,000-sample session bootstrap (seed 0), with the same search at rest."""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
D, REPO, OUT = sys.argv[1:4]; os.makedirs(os.path.join(OUT, "S3_bin_size_1ms"), exist_ok=True)
LM = pd.read_csv(os.path.join(REPO, "09_claude_science_handoff/assets/tables/phase_landmarks.csv")); rows, S = [], {}
for f in sorted(glob.glob(os.path.join(D, "onems/res_*.npz"))):
    z = np.load(f, allow_pickle=True); keys = sorted(set(x.split("__")[0] for x in z.files if "__f1" in x)); curves = []
    for k in keys:
        s, st = map(int, k.split("_")); x = z[k + "__f1"].astype(float); ft = gaussian_filter1d(x, 5)
        for loc, ref in (("reach", 4000), ("rest", 2000), ("rest", 1500)):
            w = np.arange(ref - 600, ref + 1001); tt = int(w[ft[w].argmin()] - ref); rows.append(dict(session=s, loc=loc, t=tt, hit=int(-200 <= tt <= 0)))
        b = x[3000:3600].mean(); curves.append((x - b) / np.abs(x[3600:4800] - b).max())
    S[int(os.path.basename(f)[4:10])] = np.mean(curves, 0)
d = pd.DataFrame(rows); H = d.groupby(["session", "loc"])["hit"].mean().unstack(); x = H["reach"] - H["rest"]
M = np.array(list(S.values()))
def tmin(A, ref=4000):
    m = gaussian_filter1d(np.median(A, 0), 5); w = np.arange(ref - 200, ref + 101); return int(w[m[w].argmin()] - ref)
rng = np.random.default_rng(0); boot = np.array([tmin(M[rng.integers(0, len(M), len(M))]) for _ in range(2000)])
rest = np.array([tmin(M[rng.integers(0, len(M), len(M))], 2000) for _ in range(2000)])
loo = [tmin(np.delete(M, i, 0)) for i in range(len(M))]
out = dict(per_condition_hit_reach=float(d[d["loc"] == "reach"].hit.mean()), per_condition_hit_rest=float(d[d["loc"] == "rest"].hit.mean()),
           sessions_reach_gt_rest=int((x > 0).sum()), sessions_tied=int((x == 0).sum()), p_wilcoxon=float(wilcoxon(x).pvalue),
           median_trough_ms_reach=float(d[d["loc"] == "reach"].t.median()), group_median_min_ms=tmin(M), loo_range_ms=[min(loo), max(loo)],
           boot_within30_of_pv=float(np.mean(np.abs(boot) <= 30)), boot_ci95_ms=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
           rest_boot_within30=float(np.mean(np.abs(rest) <= 30)))
json.dump(out, open(os.path.join(OUT, "S3_bin_size_1ms", "onems_trough_timing.json"), "w"), indent=1); print(json.dumps(out, indent=1))
