"""E14 step 1: raw spike times -> compact per-session 1-ms arrays for our Purkinje cells, with the data gates.
usage: python extract_spiketimes.py <raw_dir> <spikes_v2.npz> <out_dir> [sessions=all]

Our PCs = units with a non-empty cellID (as in 08_exclusion_applied/code/reextract.py, which maps raw -> bin10 exactly), in file order.
Gates (written to <out_dir>/gates_<sessions>.csv; a session that fails G1-G3 is NOT used downstream):
  G1  re-binning cellData/spikeTimes at 10 ms (integer 30 kHz sample arithmetic, bin = sample // 300) reproduces Bin10 counts exactly (every PC unit, every bin)
  G2  1-ms bins built from spikeTimes on edges nested in the 10-ms grid sum to Bin10 exactly. (Bin1 in the files is NOT nested: it is 3 samples
      shorter and disagrees with Bin10 in about 0.05% of 10-ms bins on 210421, so it is not used; the mismatch is recorded as info_*.)
  G3  per-reach 10-ms counts cut exactly as reextract.py (rows 100:800) equal C_k in spikes_v2.npz (same units, same reaches)
Output <out_dir>/<date>.npz:  c1 (R,7000,N) uint8 1-ms counts for ALL reaches in the file, rows 100:800 of the reach segment,
  stim, exclude, reachnum (1-based), idx_cross (bin of 1-cm crossing in the 700-bin window, 10-ms units, NaN if none), idx_end (bin of max x after crossing),
  unit_index (raw unit numbers), clusterID, depth, channel."""
import sys, os, re, glob, h5py, numpy as np, csv

def deref(f, r):
    o = f[r]
    if o.attrs.get("MATLAB_empty", 0) or (isinstance(o, h5py.Dataset) and o.attrs.get("MATLAB_class", b"") == b"canonical empty"):
        return None
    return o[()]

def process(path, z, out_dir):
    m = re.match(r"Dylan_(\d{6})_([A-Za-z0-9]+)_NPresults", os.path.basename(path)); date, mouse = m.groups()
    f = h5py.File(path, "r"); cd, rs = f["cellData"], f["ReachS"]
    ref = lambda g, k: g[k][()].ravel()
    nunit = len(ref(cd, "cellID"))
    is_pc = np.array([deref(f, r) is not None for r in ref(cd, "cellID")])
    pcs = np.nonzero(is_pc)[0]
    g = dict(session=date, mouse=mouse, n_units=nunit, n_pc=len(pcs))
    # --- gates G1, G2 on the full recording. Binning rule (diagnose_ties.py): spike times lie on a 30 kHz sample grid; bin = sample // 300 (10 ms),
    # sample // 30 (1 ms). This reproduces Bin10 exactly in every session; using the file's own time vector t as bin edges does NOT (t carries
    # accumulated float error that flips ties for spikes exactly on a 10-ms boundary in 6/21 sessions). Integer arithmetic also nests 1 ms in 10 ms exactly.
    b10 = [np.asarray(deref(f, r)) for r in ref(cd, "Bin10")]
    t = b10[pcs[0]][0]; dt = 0.01; n = len(t)
    counts10 = np.stack([b10[k][1] for k in pcs], 1)                       # (T10, N)
    edges_file = np.concatenate([t, [t[-1] + dt]]); b1 = []
    bad1 = 0; n1 = 0; bad2 = 0; n2 = 0; max1 = 0; legacy = 0; binfo = 0; grid_err = 0.0; outside = 0
    for j, k in enumerate(pcs):
        st = np.asarray(deref(f, ref(cd, "spikeTimes")[k])).ravel()
        q = np.rint(st * 30000.0); grid_err = max(grid_err, float(np.abs(st * 30000.0 - q).max())); sm = q.astype(np.int64)
        i10 = sm // 300; i1 = sm // 30
        outside += int(((i10 < 0) | (i10 >= n)).sum())
        h10 = np.bincount(i10[(i10 >= 0) & (i10 < n)], minlength=n)[:n]
        h1 = np.bincount(i1[(i1 >= 0) & (i1 < n * 10)], minlength=n * 10)[:n * 10]
        max1 = max(max1, int(h1.max())); b1.append(h1.astype(np.uint8))
        bad1 += int((h10 != counts10[:, j]).sum()); n1 += n                    # G1
        bad2 += int((h1.reshape(n, 10).sum(1) != h10).sum()); n2 += n          # G2 (nesting)
        hl, _ = np.histogram(st, bins=edges_file); legacy += int((hl != counts10[:, j]).sum())   # info: the file-edge rule that failed in 6 sessions
        a1 = np.asarray(deref(f, ref(cd, "Bin1")[k]))[1]; L = min(len(a1), len(h1)); binfo += int((a1[:L].astype(int) != h1[:L]).sum())
    g.update(G1_bins=n1, G1_mismatch=bad1, G2_bins=n2, G2_mismatch=bad2, Bin10_len=n, max_count_1ms=max1, spikes_outside_grid=outside,
             info_spiketimes_max_off_30kHz_grid=grid_err, info_legacy_fileedge_G1_mismatch=legacy, info_Bin1_len=int(len(a1)),
             info_Bin1_vs_spikeTimes_1ms_mismatch=binfo, info_Bin1_total=int(L) * len(pcs))
    # --- reaches, cut as reextract.py
    stim = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "stim")]
    excl = [None if (v := deref(f, r)) is None else int(np.asarray(v).ravel()[0]) for r in ref(rs, "exclude")]
    fk = [deref(f, r) for r in ref(rs, "filt_kin")]
    R = len(stim); N = len(pcs)
    c1 = np.zeros((R, 7000, N), np.uint8); idx_cross = np.full(R, np.nan); idx_end = np.full(R, np.nan)
    bad3 = 0; n3 = 0; ok_keys = 0
    for i in range(R):
        K = np.asarray(fk[i]).T; tk = K[:, 0]; x = K[:, 1]
        mask = (t > tk[0]) & (t < tk[-1]); idx = np.nonzero(mask)[0]
        if len(idx) < 800: continue
        sel = idx[100:800]                                                    # 700 10-ms bins; must be consecutive
        assert (np.diff(sel) == 1).all()
        a0 = sel[0] * 10
        for j in range(N): c1[i, :, j] = b1[j][a0:a0 + 7000] if a0 + 7000 <= len(b1[j]) else 0
        tt = t[sel]; xr = np.interp(tt, tk, x)
        cr = np.argmax(xr > 1) if (xr > 1).any() else -1
        if cr >= 0:
            idx_cross[i] = cr; idx_end[i] = cr + int(np.argmax(xr[cr:]))
        # G3
        key = "C_%s_%s" % (date, stim[i])
        if key in z and stim[i] is not None:
            rn = z["reachnum_" + key[2:]] if False else z["reachnum_%s_%s" % (date, stim[i])]
            where = np.nonzero(rn == i + 1)[0]
            if len(where):
                Cz = z[key][where[0]]                                         # (700, N)
                mine = counts10[sel]
                if Cz.shape == mine.shape: bad3 += int((Cz != mine).sum()); n3 += mine.size; ok_keys += 1
                else: bad3 += mine.size; n3 += mine.size
    g.update(G3_reaches_checked=ok_keys, G3_cells=n3, G3_mismatch=bad3, n_reaches=R,
             N_spikes_v2=int(z["X_%s_0" % date].shape[2]), gate_pass=bool(bad1 == 0 and bad2 == 0 and bad3 == 0 and ok_keys > 0 and N == z["X_%s_0" % date].shape[2]))
    num = lambda k: np.array([np.nan if (v := deref(f, r)) is None else float(np.asarray(v).ravel()[0]) for r in ref(cd, k)])
    np.savez_compressed(os.path.join(out_dir, date + ".npz"), c1=c1, stim=np.array([-1 if s is None else s for s in stim]),
                        exclude=np.array([-1 if e is None else e for e in excl]), reachnum=np.arange(1, R + 1), idx_cross=idx_cross, idx_end=idx_end,
                        unit_index=pcs, clusterID=num("clusterID")[pcs], depth=num("depth")[pcs], channel=num("channel")[pcs], mouse=mouse)
    f.close(); return g

if __name__ == "__main__":
    raw, npz, out = sys.argv[1:4]; sel = None if len(sys.argv) < 5 or sys.argv[4] == "all" else sys.argv[4].split(",")
    os.makedirs(out, exist_ok=True); z = np.load(npz, allow_pickle=True); rows = []
    for p in sorted(glob.glob(os.path.join(raw, "Dylan_*_NPresults*.mat"))):
        date = re.match(r"Dylan_(\d{6})", os.path.basename(p)).group(1)
        if sel is not None and date not in sel: continue
        g = process(p, z, out); rows.append(g); print(g, flush=True)
    with open(os.path.join(out, "gates_%s.csv" % ("all" if sel is None else "_".join(sel))), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
