"""E12 step: per-reach unscaled hand kinematics (cm) on the neural 10-ms grid, rows 100:800, from the raw NPresults v7.3 files.
Cluster-only (needs the raw files copied under $HOME; the container cannot see /data001). Reach matching follows
09_claude_science_handoff/code/reextract.py: reach number = 1-based index in ReachS; kinematics (ReachS.filt_kin, columns 1..3 of the
n x 8 array, column 0 = time) are interpolated onto the Bin10 time grid restricted to (kin_t0, kin_t1), then rows 100:800 are kept.
usage: python extract_kinematics.py <raw_dir> <out.npz>   Output keys per session date: reachnum, stim, exclude, kin (R,700,3), t0 (R,)"""
import sys, glob, os, re, numpy as np, h5py

USED = {"210421_fChR2", "210422_fChR2", "210423_fChR2", "210425_fChR2", "210511_fChR5", "210512_fChR5", "210514_fChR2",
        "210515_fChR5", "210606_fChR4", "210608_fChR4", "210614_fChR4", "210619_cChR1", "210620_cChR1", "210622_cChR1",
        "210623_cChR1", "220515_DJC002", "220516_DJC000", "220517_DJC002", "220518_DJC000", "220519_DJC002", "220520_DJC000"}

def deref(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_empty", 0) or (isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"canonical empty"):
        return None
    return o[()]

out = {}
for src in sorted(glob.glob(os.path.join(sys.argv[1], "Dylan_*_NPresults*.mat"))):
    date, mouse = re.match(r"Dylan_(\d{6})_([A-Za-z0-9]+)_NPresults", os.path.basename(src)).groups()
    if f"{date}_{mouse}" not in USED: continue
    with h5py.File(src, "r") as f:
        cd, rs = f["cellData"], f["ReachS"]
        t = deref(f, cd["Bin10"][()].ravel()[0])[0]
        ref = lambda g, k: g[k][()].ravel()
        stim = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "stim")]
        excl = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "exclude")]
        fk = [deref(f, r) for r in ref(rs, "filt_kin")]
        R = len(stim); kin = np.full((R, 700, 3), np.nan, np.float32); t0 = np.full(R, np.nan)
        for i in range(R):
            K = np.asarray(fk[i]).T; tk = K[:, 0]
            tt = t[(t > tk[0]) & (t < tk[-1])]
            if len(tt) < 800: continue
            kin[i] = np.column_stack([np.interp(tt, tk, K[:, c]) for c in (1, 2, 3)])[100:800]
            t0[i] = tt[100]
        out[f"{date}__reachnum"] = np.arange(1, R + 1); out[f"{date}__stim"] = np.array([-1 if s is None else s for s in stim])
        out[f"{date}__exclude"] = np.array([-1 if e is None else e for e in excl]); out[f"{date}__kin"] = kin; out[f"{date}__t0"] = t0
        print(date, mouse, R, flush=True)
np.savez_compressed(sys.argv[2], **out)
