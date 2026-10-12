
import sys, os, glob, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/code")
import io, contextlib
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
def cond_dirs():
    return sorted(glob.glob(os.path.join(ROOT, "*_results", "*_rep1", "full_reach", "stim_*")))
def load_spikes(d):
    df = pd.read_csv(os.path.join(d, "per_reach_state.csv")); sc = [c for c in df.columns if c.startswith("spike_n")]
    return df, df[sc].values.astype(int), df.reach_idx.values
def tag(d):
    p = d.split(os.sep); return p[-4].split("_")[0], int(p[-1].split("_")[1])
def quiet_fit(X, seed, n_cpus=1, max_iter=75):
    from model import fit_ising_model
    with contextlib.redirect_stdout(io.StringIO()):
        mult, _ = fit_ising_model(X, sample_size=10000, n_cpus=n_cpus, max_iter=max_iter, eta=0.005, rng_seed=seed)
    return np.asarray(mult)
