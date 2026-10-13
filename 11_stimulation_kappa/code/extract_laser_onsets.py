"""Step 1a: dump per-reach ReachS fields (stim, stimtime, stimtloc, exclude, qm) from the raw v7.3 files.
usage: python extract_laser_onsets.py <raw_dir> <out_csv>
stimtime is written with 17 significant digits (run 1 used %.6g, which truncated absolute times above 1000 s to 10 ms: amendment A3).
Output: one row per (session, reach index in ReachS, 0-based): stim code, exclude flag, qm, stimtime values, stimtloc values."""
import sys, os, glob, re, h5py, numpy as np, csv

def val(f, ref):
    return np.array(f[ref]).ravel()

def main(raw_dir, out_csv):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "Dylan_*_NPresults*.mat"))):
        m = re.match(r"Dylan_(\d{6})_([A-Za-z0-9]+)_NPresults", os.path.basename(path))
        date, mouse = m.group(1), m.group(2)
        with h5py.File(path, "r") as f:
            R = f["ReachS"]; n = R["stim"].shape[0]
            for i in range(n):
                g = lambda k: val(f, R[k][i, 0])
                st, stt, stl = g("stimtime"), g("stimtloc"), g("stim")
                rows.append(dict(date=date, mouse=mouse, reach0=i, stim=float(stl[0]) if len(stl) else np.nan,
                                 exclude=float(g("exclude")[0]), qm=float(g("qm")[0]),
                                 stimtime=";".join("%.17g" % x for x in st), stimtloc=";".join("%.17g" % x for x in g("stimtloc"))))
        print(date, mouse, "reaches", n, flush=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out_csv, len(rows))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
