
import glob, os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
files = sorted(glob.glob(os.path.join(ROOT, "*_results", "*_rep1", "full_reach", "stim_*", "per_reach_state.csv")))
def one(fp):
    parts = fp.split(os.sep); session = parts[-5].replace("_results",""); stim = int(parts[-2].replace("stim_",""))
    d = pd.read_csv(fp); sc = [c for c in d.columns if c.startswith("spike_n")]; N = len(sc)
    S = 2*d[sc].values.astype(np.int8) - 1
    Jrow = pd.read_csv(os.path.join(os.path.dirname(fp), "J_parameters.csv")).iloc[0]
    J = np.zeros((N,N))
    for k,v in Jrow.items():
        _, i, j = k.split("_"); J[int(i)-1, int(j)-1] = v
    iu = np.triu_indices(N, 1); Jv = J[iu]
    SiSj_a = S[:, iu[0]]; SiSj_b = S[:, iu[1]]
    coact = (SiSj_a==1)&(SiSj_b==1); cosil = (SiSj_a==-1)&(SiSj_b==-1); disc = ~(coact|cosil)
    jA = (coact*Jv).sum(1); jS = (cosil*Jv).sum(1); jD = -(disc*Jv).sum(1)
    out = d[["reach_idx","x","y","z","firing_rate","energy","j","h"]].copy()
    out["t"] = out.groupby("reach_idx").cumcount()
    out["j_coact"] = jA; out["j_cosil"] = jS; out["j_disc"] = jD
    out["n_coact_pairs"] = coact.sum(1); out["n_cosil_pairs"] = cosil.sum(1)
    out["session"] = session; out["stim"] = stim; out["n_neurons"] = N
    err = np.abs(out.j - (jA+jS+jD)).max()
    return out, dict(session=session, stim=stim, N=N, n_pairs=len(Jv), max_abs_err=err, J_pos_frac=(Jv>0).mean(), J_mean=Jv.mean())
res = Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))).map(one, files)
pd.concat([r[0] for r in res], ignore_index=True).to_csv("per_reach_cosilence.csv.gz", index=False, compression="gzip")
pd.DataFrame([r[1] for r in res]).to_csv("cosilence_check.csv", index=False)
print(len(files), max(r[1]["max_abs_err"] for r in res))
