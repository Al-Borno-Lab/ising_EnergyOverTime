
import glob, os, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
files = sorted(glob.glob(os.path.join(ROOT, "*_results", "*_rep1", "full_reach", "stim_*", "per_reach_state.csv")))
T = 700
def one(fp):
    p = fp.split(os.sep); s = p[-5].split("_")[0]; st = int(p[-2].split("_")[1])
    d = pd.read_csv(fp); sc = [c for c in d.columns if c.startswith("spike_n")]; N = len(sc)
    S = (2 * d[sc].values - 1).astype(float).reshape(-1, T, N)          # reaches x T x N
    Jr = pd.read_csv(os.path.join(os.path.dirname(fp), "J_parameters.csv")).iloc[0]
    J = np.zeros((N, N))
    for k, v in Jr.items():
        _, i, j = k.split("_"); J[int(i) - 1, int(j) - 1] = v
    m = S.mean(0)                                                          # T x N  (trial-averaged spin = rate)
    j_obs = np.einsum("rti,ij,rtj->t", S, J, S) / S.shape[0]              # mean over reaches of sum_{i<j} J s s
    j_rate = np.einsum("ti,ij,tj->t", m, J, m)                             # prediction from per-neuron rates alone (independence)
    # leave-one-reach-out rate prediction removes the 1/R self-term bias
    R = S.shape[0]
    return pd.DataFrame(dict(session=s, stim=st, t=np.arange(T), N=N, R=R, j_obs=j_obs, j_rate=j_rate, j_cov=j_obs - j_rate,
                             fr=(m.mean(1) + 1) / 2))
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    out = pd.concat(p.map(one, files))
out.to_csv("j_rate_vs_cov.csv.gz", index=False, compression="gzip"); print(out.shape)
