"""Write copies of the *_bin10_spike.mat files with ReachS.exclude==1 reaches removed. Originals untouched."""
import glob, os, sys, numpy as np, pandas as pd, scipy.io as sio
sys.path.insert(0, "/home/enserrog/ising_EnergyOverTime/energy_over_time_script")
from utils import loadmat as proj_loadmat
SRC = "/data001/projects/enserrog/AbigailData/energy_over_time"
OUT = "/data001/projects/enserrog/AbigailData/energy_over_time_exclude_applied"
os.makedirs(OUT, exist_ok=True)
RR = pd.read_csv("reextract_reaches.csv")
rows = []
for p in sorted(glob.glob(f"{SRC}/*_bin10_spike.mat")):
    ident = os.path.basename(p)[:-len("_bin10_spike.mat")]
    ex = set(RR[(RR.ident == ident) & (RR.exclude == 1)].reach.astype(int))
    ka = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["kinAggrogate"]
    new = {}
    for b in ka._fieldnames:
        blk = getattr(ka, b); ks = sorted({int(n.split("_")[1]) for n in blk._fieldnames if n.startswith("n_")})
        keep = [k for k in ks if k not in ex]
        d = {}
        for k in keep:
            for pre in ("l", "n", "label"):
                if f"{pre}_{k}" in blk._fieldnames: d[f"{pre}_{k}"] = np.asarray(getattr(blk, f"{pre}_{k}"))
        new[b] = d
        rows.append(dict(ident=ident, block=b, n_before=len(ks), n_after=len(keep), dropped=sorted(set(ks) & ex)))
    q = f"{OUT}/{os.path.basename(p)}"; sio.savemat(q, {"kinAggrogate": new}, do_compression=True)
    # reload with the project's own loader and check the counts
    chk = proj_loadmat(q)["kinAggrogate"]
    for b in new:
        got = sum(1 for n in chk[b] if str(n).startswith("n_")); assert got == len([n for n in new[b] if n.startswith("n_")]), (ident, b, got)
        k0 = sorted(int(n.split("_")[1]) for n in chk[b] if str(n).startswith("n_"))[0]
        assert np.array_equal(np.asarray(chk[b][f"n_{k0}"]), new[b][f"n_{k0}"]), (ident, b)
    print(ident, "ok", flush=True)
pd.DataFrame(rows).to_csv("exclusion_filter_log.csv", index=False)
print("done")
