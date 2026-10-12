
import glob, os, pandas as pd
root = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
files = sorted(glob.glob(os.path.join(root, "*_results", "*_rep1", "full_reach", "stim_*", "per_reach_state.csv")))
out = []
for fp in files:
    parts = fp.split(os.sep)
    session = parts[-5].replace("_results", ""); stim = int(parts[-2].replace("stim_", ""))
    df = pd.read_csv(fp)
    nn = sum(c.startswith("spike_n") for c in df.columns)
    d = df[["reach_idx","x","y","z","firing_rate","energy","j","h"]].copy()
    d["t"] = d.groupby("reach_idx").cumcount()
    d["session"] = session; d["stim"] = stim; d["n_neurons"] = nn
    out.append(d)
all_ = pd.concat(out, ignore_index=True)
all_.to_csv("per_reach_compact.csv.gz", index=False, compression="gzip")
print(len(files), all_.shape)
