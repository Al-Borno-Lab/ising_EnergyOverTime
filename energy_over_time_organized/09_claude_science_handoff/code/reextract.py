"""Independent Python re-implementation of organizeData.m on Dylan's NPresults source files,
compared against the *_bin10_spike.mat files the pipeline used. Also exports per-reach and per-unit tables."""
import glob, os, re, json, sys, numpy as np, pandas as pd, h5py, scipy.io as sio
SRC = "/data001/projects/enserrog/dylan/AbigailUpdated"
B10 = "/data001/projects/enserrog/AbigailData/energy_over_time"

def deref(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_empty", 0) or (isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"canonical empty"):
        return None
    return o[()]

def strval(f, r):
    """MATLAB string objects are MCOS; return a hashable signature (non-empty marker + raw ints)."""
    v = deref(f, r)
    return None if v is None else tuple(np.asarray(v).ravel().tolist())

def rescale(x, lo=0.0, hi=1.0):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.full_like(x, lo) if mx == mn else lo + (x - mn) / (mx - mn) * (hi - lo)

b10_files = {os.path.basename(p)[:-len("_bin10_spike.mat")]: p for p in glob.glob(f"{B10}/*_bin10_spike.mat")}
sess_rows, reach_rows, unit_rows, cmp_rows = [], [], [], []
for src in sorted(glob.glob(f"{SRC}/Dylan_*.mat")):
    m = re.match(r"Dylan_(\d{6})_([A-Za-z0-9]+)_NPresults", os.path.basename(src)); date, mouse = m.groups(); ident = f"{date}_{mouse}"
    try:
        f = h5py.File(src, "r")
    except Exception as e:
        sess_rows.append(dict(source=os.path.basename(src), error=str(e))); continue
    cd, rs = f["cellData"], f["ReachS"]
    ncell = cd["Bin10"].shape[0] * cd["Bin10"].shape[1]
    ref = lambda g, k: g[k][()].ravel()
    ids = [strval(f, r) for r in ref(cd, "cellID")]
    # MATLAB string payload: last element encodes the string object index; non-empty = labelled (PC in organizeData)
    is_pc = np.array([x is not None for x in ids])
    num = lambda g, k: [None if (v := deref(f, r)) is None else float(np.asarray(v).ravel()[0]) for r in ref(g, k)]
    ch, dp, cv2, mad, fr, clu = (num(cd, k) for k in ["channel", "depth", "CV2", "MAD", "fr", "clusterID"])
    has_cs = [deref(f, r) is not None for r in ref(cd, "CS_Bin1")] if "CS_Bin1" in cd else [None] * len(ids)
    for k in range(len(ids)):
        unit_rows.append(dict(ident=ident, unit=k, labelled=bool(is_pc[k]), channel=ch[k], depth=dp[k], CV2=cv2[k], MAD=mad[k], fr=fr[k], clusterID=clu[k], has_CS=has_cs[k]))
    bins = [deref(f, r) for r in ref(cd, "Bin10")]            # each (2, T): row0 time, row1 count
    t = bins[0][0]
    pcs = [k for k in range(len(ids)) if is_pc[k]]
    counts = np.stack([bins[k][1] for k in pcs], 1) if pcs else np.zeros((len(t), 0))
    if "stim" not in rs:
        sess_rows.append(dict(source=os.path.basename(src), ident=ident, in_b10=ident in b10_files, n_units=len(ids), n_labelled=int(is_pc.sum()),
                              error="ReachS has no 'stim' field; fields=" + ",".join(rs.keys())))
        f.close(); print(ident, "NO STIM FIELD", flush=True); continue
    stim = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "stim")]
    excl = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "exclude")]
    qm = num(rs, "qm")
    st_t = [deref(f, r) for r in ref(rs, "stimtime")] if "stimtime" in rs else [None] * len(stim)
    fk = [deref(f, r) for r in ref(rs, "filt_kin")]            # (8, n) -> MATLAB n x 8
    rk = [deref(f, r) for r in ref(rs, "real_kin")] if "real_kin" in rs else [None] * len(stim)
    # comparison file
    b = None
    if ident in b10_files:
        b = sio.loadmat(b10_files[ident], squeeze_me=True, struct_as_record=False)["kinAggrogate"]
    for i in range(len(stim)):
        K = np.asarray(fk[i]).T                                 # n x 8
        tk = K[:, 0]; x = K[:, 1]
        mask = (t > tk[0]) & (t < tk[-1]); tt = t[mask]
        kin = np.column_stack([tt] + [rescale(np.interp(tt, tk, K[:, c]), *((0, 1) if c <= 4 else (-1, 1))) for c in range(1, K.shape[1])])
        n = counts[mask]
        # unscaled kinematics on the neural time grid
        xr = np.interp(tt, tk, x)
        cross = int(np.argmax(xr > 1)) if (xr > 1).any() else -1
        vx = np.gradient(xr, tt)
        win = slice(100, 800)
        pv_in_win = int(np.argmax(vx[100:800])) if len(vx) >= 800 else -1
        stv = None if st_t[i] is None else np.asarray(st_t[i]).ravel().tolist()[:3]
        rr = dict(ident=ident, date=date, mouse=mouse, reach=i + 1, stim=stim[i], exclude=excl[i], qm=qm[i], n_bins=int(mask.sum()),
                  kin_t0=float(tk[0]), kin_t1=float(tk[-1]), kin_n=len(tk), kin_dt=float(np.median(np.diff(tk))),
                  x_min_cm=float(np.nanmin(x)), x_max_cm=float(np.nanmax(x)), idx_1cm=cross, idx_1cm_in_window=cross - 100 if cross >= 0 else None,
                  idx_peakvel_in_window=pv_in_win, peak_vx_cm_s=float(np.nanmax(vx[100:800])) if len(vx) >= 800 else None,
                  stimtime=stv, n_bins_ge2=int((n[win] >= 2).sum()) if len(n) >= 800 else None, n_bins_ge1=int((n[win] >= 1).sum()) if len(n) >= 800 else None)
        # compare with bin10_spike file
        if b is not None and stim[i] is not None:
            blk = getattr(b, f"stim_{stim[i]}", None)
            if blk is not None and f"n_{i+1}" in blk._fieldnames:
                bn = np.asarray(getattr(blk, f"n_{i+1}"), float); bl = np.asarray(getattr(blk, f"l_{i+1}"), float)
                rr.update(in_b10=True, b10_rows=bn.shape[0], b10_cols=bn.shape[1] - 1, rows_match=bn.shape[0] == len(tt),
                          n_match=bool(bn.shape == np.column_stack([tt, n]).shape and np.allclose(bn[:, 1:], n)),
                          time_match=bool(bn.shape[0] == len(tt) and np.allclose(bn[:, 0], tt)),
                          kin_maxdiff=float(np.nanmax(np.abs(bl - kin))) if bl.shape == kin.shape else None,
                          excl_field=float(np.asarray(getattr(blk, f"exclude_{i+1}"))) if f"exclude_{i+1}" in blk._fieldnames else None)
            else:
                rr.update(in_b10=False)
        reach_rows.append(rr)
    sess_rows.append(dict(source=os.path.basename(src), ident=ident, in_b10=ident in b10_files, n_units=len(ids), n_labelled=int(is_pc.sum()),
                          b10_N=(np.asarray(getattr(getattr(b, b._fieldnames[0]), [x for x in getattr(b, b._fieldnames[0])._fieldnames if x.startswith("n_")][0])).shape[1] - 1) if b is not None else None,
                          n_reaches=len(stim), n_stim_empty=sum(s is None for s in stim), n_excluded=sum(e == 1 for e in excl if e is not None),
                          n_counts_ge2_total=int((counts >= 2).sum()), max_count=int(counts.max()) if counts.size else None,
                          PClayerdepth=float(np.asarray(f["PClayerdepth"][()]).ravel()[0]) if "PClayerdepth" in f else None))
    f.close(); print(ident, "done", flush=True)
pd.DataFrame(sess_rows).to_csv("reextract_sessions.csv", index=False)
pd.DataFrame(reach_rows).to_csv("reextract_reaches.csv", index=False)
pd.DataFrame(unit_rows).to_csv("reextract_units.csv", index=False)
print(sorted(b10_files))
