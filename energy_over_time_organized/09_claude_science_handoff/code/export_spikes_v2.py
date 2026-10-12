"""Export binarized spikes (rows 100:800, counts>0) for all 63 conditions from the original bin10_spike files,
with the original 5-fold reach assignment (rng(stim).permutation over reach order) and Dylan's exclude flag per reach.
Also exports unscaled x (cm) on the same bins and the stimulation time from the source ReachS (via reextract_reaches.csv)."""
import glob, os, numpy as np, pandas as pd, scipy.io as sio
B10 = "/data001/projects/enserrog/AbigailData/energy_over_time"
RR = pd.read_csv("reextract_reaches.csv")
out = {}; rows = []
for p in sorted(glob.glob(f"{B10}/*_bin10_spike.mat")):
    ident = os.path.basename(p)[:-len("_bin10_spike.mat")]; s = ident[:6]
    ka = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["kinAggrogate"]
    for b in ka._fieldnames:
        st = int(b.split("_")[1]); blk = getattr(ka, b)
        ks = sorted(int(n.split("_")[1]) for n in blk._fieldnames if n.startswith("n_"))
        X = np.stack([(np.asarray(getattr(blk, f"n_{k}"), float)[100:800, 1:] > 0) for k in ks]).astype(np.uint8)
        C = np.stack([np.asarray(getattr(blk, f"n_{k}"), float)[100:800, 1:] for k in ks]).astype(np.uint8)
        T0 = np.stack([np.asarray(getattr(blk, f"n_{k}"), float)[100:800, 0] for k in ks])
        R = len(ks); rng = np.random.default_rng(st); folds = np.array_split(rng.permutation(np.arange(R)), 5)
        fold = np.empty(R, int)
        for f, te in enumerate(folds): fold[te] = f
        ex = RR.set_index(["ident", "reach"]).exclude.reindex([(ident, k) for k in ks]).values.astype(int)
        key = f"{s}_{st}"
        out[f"X_{key}"] = X; out[f"C_{key}"] = C; out[f"t_{key}"] = T0; out[f"reachnum_{key}"] = np.array(ks); out[f"fold_{key}"] = fold; out[f"exclude_{key}"] = ex
        rows.append(dict(session=s, ident=ident, stim=st, R=R, N=X.shape[2], n_excluded=int(ex.sum())))
np.savez_compressed("spikes_v2.npz", **out)
pd.DataFrame(rows).to_csv("spikes_v2_index.csv", index=False); print(len(rows))
