
from ising_common import *
from multiprocessing import Pool
def task(d):
    s, st = tag(d); df, X, reach = load_spikes(d); N = X.shape[1]
    C = np.corrcoef(X.T); np.fill_diagonal(C, -1); i, j = np.unravel_index(np.nanargmax(C), C.shape)
    drop = i if X[:, i].mean() < X[:, j].mean() else j
    keep = [k for k in range(N) if k != drop]; Xk = X[:, keep]; Nk = len(keep)
    m = quiet_fit(Xk, seed=1); h = m[:Nk]; J = np.zeros((Nk, Nk)); iu = np.triu_indices(Nk, 1); J[iu] = m[Nk:]
    S = 2.0 * Xk - 1; hterm = S @ h; jterm = np.einsum("ti,ij,tj->t", S, J, S)
    out = df[["reach_idx", "x", "y", "z", "firing_rate", "energy", "j", "h"]].copy()
    out["t"] = out.groupby("reach_idx").cumcount(); out["j_dedup"] = jterm; out["h_dedup"] = hterm; out["energy_dedup"] = -(jterm + hterm)
    out["firing_rate_dedup"] = Xk.mean(1); out["session"] = s; out["stim"] = st; out["n_neurons"] = Nk
    return out, dict(session=s, stim=st, N=N, pair_i=int(i), pair_j=int(j), pair_r=float(C[i, j]), dropped=int(drop),
                     rate_i=float(X[:, i].mean()), rate_j=float(X[:, j].mean()), second_max_r=float(np.sort(C[np.triu_indices(N, 1)])[-2]))
dirs = [d for d in cond_dirs() if tag(d)[0] in ("210620", "220515")]
with Pool(6) as p: res = p.map(task, dirs)
pd.concat([r[0] for r in res]).to_csv("dedup_per_bin.csv.gz", index=False, compression="gzip")
pd.DataFrame([r[1] for r in res]).to_csv("dedup_units.csv", index=False)
