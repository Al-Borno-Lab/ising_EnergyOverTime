"""Print the HDF5 structure of ReachS (and the stim-related fields) in one raw v7.3 file. usage: python inspect_reachs.py <file.mat>"""
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
print("top-level keys:", list(f.keys()))
def show(g, name, depth=0):
    for k in g.keys():
        o = g[k]
        if isinstance(o, h5py.Dataset):
            print("  " * depth + f"{name}/{k}: dataset shape={o.shape} dtype={o.dtype}")
        else:
            print("  " * depth + f"{name}/{k}: group")
            if depth < 1: show(o, f"{name}/{k}", depth + 1)
show(f["ReachS"], "ReachS")
R = f["ReachS"]
n = R["stimtime"].shape[0]
for key in ("stim", "stimtime", "stimtloc", "exclude", "qm"):
    d = R[key]; print("==", key, d.shape)
    for i in list(range(4)) + [n - 1]:
        o = f[d[i, 0]]; a = np.array(o)
        print("  reach", i, "shape", a.shape, "values", a.ravel()[:8])
