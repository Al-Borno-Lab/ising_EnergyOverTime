
from ising_common import *
from multiprocessing import Pool
K = 5
def one(d):
    s, st = tag(d); df, X, reach = load_spikes(d); N = X.shape[1]
    ureach = np.unique(reach); rng = np.random.default_rng(st); folds = np.array_split(rng.permutation(ureach), K)
    fold_of = {r: f for f, test in enumerate(folds) for r in test}
    R = len(ureach); Xr = X.reshape(R, 700, N).astype(np.uint8); rid = reach.reshape(R, 700)[:, 0]
    return f"{s}_{st}", Xr, rid, np.array([fold_of[r] for r in rid]), df.firing_rate.values.reshape(R, 700).mean(0)
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    res = p.map(one, cond_dirs())
out = {}
for k, Xr, rid, fold, fr in res:
    out[f"X_{k}"] = Xr; out[f"reach_{k}"] = rid; out[f"fold_{k}"] = fold; out[f"fr_{k}"] = fr
np.savez_compressed("spikes_all.npz", **out); print(len(res))
