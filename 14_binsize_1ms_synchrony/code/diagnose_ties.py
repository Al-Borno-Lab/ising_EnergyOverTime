"""E14: which tie-breaking rule for spikes on 10-ms boundaries reproduces Bin10 exactly? usage: python diagnose_ties.py <file.mat> <out.json>
Rules for the bin index of a spike time s on the file's own grid t (t[0]=0, nominal dt=0.01 s):
 R0 file edges t, left-closed [t_j, t_j+1);  R1 edges k*0.01;  R2 edges cumsum(0.01);  R3 linspace;  R4 file edges, right-closed (t_j, t_j+1];
 R5 integer samples at 30 kHz: floor(round(s*30000)/300);  R6 floor(s/0.01 + 1e-9);  R7 ceil(s/0.01 - 1e-9) - 1;  R8 round(s/0.01) (nearest, ties to even).
Reports total mismatching bins against Bin10 over all PC units for each rule."""
import sys, json, h5py, numpy as np

def deref(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_empty", 0) or (isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"canonical empty"): return None
    return o[()]

f = h5py.File(sys.argv[1], "r"); cd = f["cellData"]; ref = lambda g, k: g[k][()].ravel()
pcs = [k for k, r in enumerate(ref(cd, "cellID")) if deref(f, r) is not None]
b10 = {k: np.asarray(deref(f, ref(cd, "Bin10")[k])) for k in pcs}; t = b10[pcs[0]][0]; n = len(t); dt = 0.01
rules = {
    "R0_file_edges_left_closed": lambda st: np.searchsorted(np.concatenate([t, [t[-1] + dt]]), st, side="right") - 1,
    "R1_k*0.01_left_closed": lambda st: np.searchsorted(np.arange(n + 1) * dt, st, side="right") - 1,
    "R2_cumsum_left_closed": lambda st: np.searchsorted(np.cumsum(np.r_[0.0, np.full(n, dt)]), st, side="right") - 1,
    "R3_linspace_left_closed": lambda st: np.searchsorted(np.linspace(0, n * dt, n + 1), st, side="right") - 1,
    "R4_file_edges_right_closed": lambda st: np.searchsorted(np.concatenate([t, [t[-1] + dt]]), st, side="left") - 1,
    "R5_int_samples_30k": lambda st: np.round(st * 30000).astype(np.int64) // 300,
    "R6_floor(s/dt+1e-9)": lambda st: np.floor(st / dt + 1e-9).astype(np.int64),
    "R7_ceil(s/dt-1e-9)-1": lambda st: np.ceil(st / dt - 1e-9).astype(np.int64) - 1,
    "R8_round(s/dt)": lambda st: np.rint(st / dt).astype(np.int64),
    "R9_floor(s/dt)": lambda st: np.floor(st / dt).astype(np.int64),
}
res = {name: 0 for name in rules}; tot_bins = 0; ties = 0
for k in pcs:
    st = np.asarray(deref(f, ref(cd, "spikeTimes")[k])).ravel(); ref10 = b10[k][1].astype(np.int64); tot_bins += n
    ties += int((np.abs(st / dt - np.rint(st / dt)) < 1e-6).sum())
    for name, fn in rules.items():
        idx = fn(st); idx = idx[(idx >= 0) & (idx < n)]
        h = np.bincount(idx, minlength=n)[:n]; res[name] += int((h != ref10).sum())
out = dict(file=sys.argv[1].split("/")[-1], n_pc=len(pcs), bins=tot_bins, spikes_on_10ms_boundary=ties, mismatches=res)
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out))
