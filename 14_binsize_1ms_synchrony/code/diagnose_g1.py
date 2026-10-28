"""E14: where do the G1 mismatches (spikeTimes histogram vs Bin10) occur, and what are they? usage: python diagnose_g1.py <file.mat> <out.json>
For each PC unit: mismatching bins, the count difference (hist - Bin10), whether a neighbouring bin has the opposite difference (a spike assigned to the
adjacent bin), the distance of the involved spikes to the bin edges, and whether the mismatch bin lies inside any analysed reach window (rows 100:800 of a reach segment)."""
import sys, json, h5py, numpy as np

def deref(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_empty", 0) or (isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"canonical empty"): return None
    return o[()]

f = h5py.File(sys.argv[1], "r"); cd, rs = f["cellData"], f["ReachS"]; ref = lambda g, k: g[k][()].ravel()
pcs = [k for k, r in enumerate(ref(cd, "cellID")) if deref(f, r) is not None]
b10 = [np.asarray(deref(f, r)) for r in ref(cd, "Bin10")]; t = b10[pcs[0]][0]; dt = 0.01
edges = np.concatenate([t, [t[-1] + dt]])
# analysed windows
inwin = np.zeros(len(t), bool)
for r in ref(rs, "filt_kin"):
    K = np.asarray(deref(f, r)).T; tk = K[:, 0]; idx = np.nonzero((t > tk[0]) & (t < tk[-1]))[0]
    if len(idx) >= 800: inwin[idx[100:800]] = True
out = dict(T=len(t), t_first=float(t[0]), t_last=float(t[-1]), dt_check=float(np.max(np.abs(np.diff(t) - dt))), units=[])
tot = 0
for j, k in enumerate(pcs):
    st = np.asarray(deref(f, ref(cd, "spikeTimes")[k])).ravel()
    h, _ = np.histogram(st, bins=edges); d = h.astype(int) - b10[k][1].astype(int)
    bad = np.nonzero(d)[0]; tot += len(bad)
    if len(bad) == 0: continue
    # spikes outside histogram range (before t[0] or at/after the last edge)
    n_out = int(((st < edges[0]) | (st >= edges[-1])).sum())
    comp = int(sum(1 for b in bad if (b + 1 < len(d) and d[b] * d[b + 1] < 0) or (b > 0 and d[b] * d[b - 1] < 0)))
    # spikes within 1e-6 s of an edge for mismatching bins
    near = 0
    for b in bad[:200]:
        lo, hi = edges[b], edges[b + 1]; s = st[(st > lo - 2e-6) & (st < hi + 2e-6)]
        near += int(((np.abs(s - lo) < 2e-6) | (np.abs(s - hi) < 2e-6)).any())
    out["units"].append(dict(unit=int(k), n_bad=len(bad), sum_diff=int(d.sum()), abs_diff_hist={int(a): int(c) for a, c in zip(*np.unique(np.abs(d[bad]), return_counts=True))},
                             in_analysed_windows=int(inwin[bad].sum()), complementary_neighbour=comp, near_edge_first200=near,
                             first_bad_times=[float(t[b]) for b in bad[:5]], spikes_outside_grid=n_out, bin10_total=int(b10[k][1].sum()), hist_total=int(h.sum()), n_spikes=int(len(st))))
out["total_bad"] = tot
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out)[:1500])
