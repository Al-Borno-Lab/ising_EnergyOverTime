
import os, glob, json
import numpy as np
import pandas as pd

DT = 0.01
WINDOW, STEP = 25, 5
SEARCH_LO, SEARCH_HI = 350, 500

def windowed_variance(ts, window=WINDOW, step=STEP):
    n = len(ts)
    starts = np.arange(0, n - window + 1, step)
    centers = starts + window // 2
    vals = np.array([np.var(ts[s:s+window]) for s in starts])
    return centers, vals

def restricted_extremum(centers, vals, lo=SEARCH_LO, hi=SEARCH_HI, mode="max"):
    mask = (centers >= lo) & (centers <= hi)
    if not mask.any():
        return np.nan, True
    sub_centers, sub_vals = centers[mask], vals[mask]
    idx = np.argmax(sub_vals) if mode == "max" else np.argmin(sub_vals)
    is_boundary = (idx == 0) or (idx == len(sub_vals) - 1)
    return sub_centers[idx], is_boundary

def per_reach_traces(df):
    reach_ids = sorted(df["reach_idx"].unique())
    C_E, C_J, C_H, VEL, ACC = [], [], [], [], []
    centers = None
    for rid in reach_ids:
        sub = df[df["reach_idx"] == rid]
        e, j, h, x = sub["energy"].values, sub["j"].values, sub["h"].values, sub["x"].values
        c, ce = windowed_variance(e)
        _, cj = windowed_variance(j)
        _, ch = windowed_variance(h)
        vel = np.gradient(x, DT)
        acc = np.gradient(vel, DT)
        centers = c
        C_E.append(ce); C_J.append(cj); C_H.append(ch)
        VEL.append(vel); ACC.append(acc)
    return centers, np.array(C_E), np.array(C_J), np.array(C_H), np.array(VEL), np.array(ACC)

root = "/home/enserrog/ising_EnergyOverTime/energy_over_time_script/outputs/energy_decomp_Jul_10_svg"
files = sorted(glob.glob(os.path.join(root, "*_results", "*_rep1", "full_reach", "stim_*", "per_reach_state.csv")))

rows = []
errors = []
for fp in files:
    parts = fp.split(os.sep)
    session = parts[-5].replace("_results", "")
    stim = int(parts[-2].replace("stim_", ""))
    try:
        df = pd.read_csv(fp)
        n_reaches = df["reach_idx"].nunique()
        centers, C_E, C_J, C_H, VEL, ACC = per_reach_traces(df)
        mean_CE, mean_CJ, mean_CH = C_E.mean(axis=0), C_J.mean(axis=0), C_H.mean(axis=0)
        mean_vel, mean_acc = VEL.mean(axis=0), ACC.mean(axis=0)

        peak_vel_idx = int(np.argmax(np.abs(mean_vel)))
        peak_acc_idx = int(np.argmax(np.abs(mean_acc)))

        pk_CE, b_pk_CE = restricted_extremum(centers, mean_CE, mode="max")
        tr_CE, b_tr_CE = restricted_extremum(centers, mean_CE, mode="min")
        pk_CJ, b_pk_CJ = restricted_extremum(centers, mean_CJ, mode="max")
        tr_CJ, b_tr_CJ = restricted_extremum(centers, mean_CJ, mode="min")
        pk_CH, b_pk_CH = restricted_extremum(centers, mean_CH, mode="max")
        tr_CH, b_tr_CH = restricted_extremum(centers, mean_CH, mode="min")

        rows.append(dict(
            session=session, stim=stim, n_reaches=n_reaches,
            peak_vel_idx=peak_vel_idx, peak_acc_idx=peak_acc_idx,
            pk_CE=pk_CE, lag_pk_CE_vel=pk_CE-peak_vel_idx, lag_pk_CE_acc=pk_CE-peak_acc_idx, boundary_pk_CE=b_pk_CE,
            tr_CE=tr_CE, lag_tr_CE_vel=tr_CE-peak_vel_idx, lag_tr_CE_acc=tr_CE-peak_acc_idx, boundary_tr_CE=b_tr_CE,
            pk_CJ=pk_CJ, lag_pk_CJ_vel=pk_CJ-peak_vel_idx, lag_pk_CJ_acc=pk_CJ-peak_acc_idx, boundary_pk_CJ=b_pk_CJ,
            tr_CJ=tr_CJ, lag_tr_CJ_vel=tr_CJ-peak_vel_idx, lag_tr_CJ_acc=tr_CJ-peak_acc_idx, boundary_tr_CJ=b_tr_CJ,
            pk_CH=pk_CH, lag_pk_CH_vel=pk_CH-peak_vel_idx, lag_pk_CH_acc=pk_CH-peak_acc_idx, boundary_pk_CH=b_pk_CH,
            tr_CH=tr_CH, lag_tr_CH_vel=tr_CH-peak_vel_idx, lag_tr_CH_acc=tr_CH-peak_acc_idx, boundary_tr_CH=b_tr_CH,
        ))
    except Exception as e:
        errors.append(dict(session=session, stim=stim, error=str(e)))

df_out = pd.DataFrame(rows)
df_out.to_csv("/tmp/criticality_timing_raw.csv", index=False)
print("rows:", len(df_out), "errors:", len(errors))
if errors:
    print(json.dumps(errors, indent=2))
print(df_out.head(10).to_string())
