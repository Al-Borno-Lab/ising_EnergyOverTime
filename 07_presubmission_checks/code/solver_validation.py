
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/code")
from model import fit_ising_model
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/01_ising_model_fitting/results"
rows=[]
for s, st in [("210421",0), ("210511",1), ("220517",0)]:
    d = f"{ROOT}/{s}_results/{s}_rep1/full_reach/stim_{st}"
    df = pd.read_csv(f"{d}/per_reach_state.csv"); sc=[c for c in df.columns if c.startswith("spike_n")]; N=len(sc)
    X = df[sc].values.astype(int)
    t=time.time(); mult, solver = fit_ising_model(X, sample_size=10000, n_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK",8)), max_iter=75, eta=0.005, rng_seed=1)
    dt=time.time()-t
    Jst = pd.read_csv(f"{d}/J_parameters.csv").iloc[0].values.astype(float); hst = pd.read_csv(f"{d}/h_parameters.csv").h_values.values
    Jn, hn = mult[N:], mult[:N]
    np.savez(f"refit_{s}_stim{st}.npz", J_stored=Jst, J_refit=Jn, h_stored=hst, h_refit=hn)
    rows.append(dict(session=s, stim=st, N=N, T=X.shape[0], fit_seconds=round(dt,1), r_J=np.corrcoef(Jst,Jn)[0,1], r_h=np.corrcoef(hst,hn)[0,1],
                     mean_absJ_stored=np.abs(Jst).mean(), mean_absJ_refit=np.abs(Jn).mean(), slope_J=np.polyfit(Jst,Jn,1)[0]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("solver_validation.csv", index=False)
