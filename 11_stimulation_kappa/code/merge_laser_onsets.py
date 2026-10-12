"""Step 1b: match ReachS.stimtime to spikes_v2 reaches and convert to bins.
usage: python merge_laser_onsets.py <reachs_raw.csv> <spikes_v2.npz> <out_csv> [dt=0.01]
Keys in spikes_v2 are '<date>_<stimcode>'; reachnum_k is 1-based into ReachS (reach0 = reachnum-1); t_k is absolute time (s).
onset_float = (stimtime - t_k[r,0]) / dt  (bin index in the 700-bin window, fractional). Only exclude_k == 0 reaches are kept."""
import sys, csv, numpy as np

def main(raw_csv, npz, out_csv, dt=0.01):
    raw = {}
    for r in csv.DictReader(open(raw_csv)):
        raw[(r["date"], int(r["reach0"]))] = r
    z = np.load(npz, allow_pickle=True)
    out = []
    for key in sorted(k[2:] for k in z.keys() if k.startswith("X_")):
        date, stim = key.split("_"); stim = int(stim)
        t, rn, ex = z["t_" + key], z["reachnum_" + key], z["exclude_" + key]
        for r in range(len(rn)):
            if ex[r] != 0: continue
            row = raw[(date, int(rn[r]) - 1)]
            assert int(float(row["stim"])) == stim, (key, r, row["stim"])
            st = [float(x) for x in row["stimtime"].split(";") if x and float(x) != 0.0]   # [0 0] is the MATLAB empty placeholder
            out.append(dict(session=date, mouse=row["mouse"], block=stim, reach_row=r, reachnum=int(rn[r]), exclude=0,
                            stimtime=st[0] if len(st) == 1 else np.nan, n_stimtime=len(st), t0=float(t[r, 0]),
                            onset_float=(st[0] - float(t[r, 0])) / dt if len(st) == 1 else np.nan))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    on = np.array([o["onset_float"] for o in out if o["block"] == 1])
    print("stim_1 reaches kept:", len(on), "with a laser time:", int(np.isfinite(on).sum()))
    print("onset_float quantiles 0/5/25/50/75/95/100:", np.nanpercentile(on, [0, 5, 25, 50, 75, 95, 100]).round(2))
    nz = [o for o in out if o["block"] != 1 and o["n_stimtime"] > 0]
    print("non-stim_1 reaches carrying a laser time:", len(nz))
    print("stim_1 reaches without exactly one laser time:", sum(1 for o in out if o["block"] == 1 and o["n_stimtime"] != 1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *(float(a) for a in sys.argv[4:5]))
