
import sys, glob, numpy as np
sys.path.insert(0, "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/code")
from utils import loadmat
rows=[]
for f in sorted(glob.glob("/home/enserrog/ising_EnergyOverTime/data/*.mat")):
    b = loadmat(f)
    for stim_key, stim in b["kinAggrogate"].items():
        for k, m in stim.items():
            m = np.array(m)
            if m.ndim==2 and (k[0]=="n" or k[:2]=="l_"):
                dt = np.diff(m[:, 0]); rows.append((f.split("/")[-1], stim_key, k[:2], m.shape[0], float(np.median(dt)), float(dt.min()), float(dt.max()), float(m[0,0]), float(m[-1,0])))
import collections
print("file stim kind n_rows median_dt min_dt max_dt t0 tend")
seen=collections.Counter()
for r in rows:
    key=(r[0], r[2])
    if seen[key] < 2: print(*r)
    seen[key]+=1
allm=np.array([r[4] for r in rows]); print("ALL median_dt values:", np.unique(np.round(allm,6))[:10], "n matrices:", len(rows))
