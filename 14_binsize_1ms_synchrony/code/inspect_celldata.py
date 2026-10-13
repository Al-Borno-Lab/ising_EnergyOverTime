"""Print the structure of cellData / ReachS in one raw v7.3 file (for E14 extraction). usage: python inspect_celldata.py <file.mat>"""
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
print("top:", list(f.keys()))
cd = f["cellData"]
for k in cd.keys():
    d = cd[k]; print("cellData/%s" % k, d.shape, d.dtype)
n = cd["Bin10"].shape[0] * cd["Bin10"].shape[1]; print("n units", n)
def get(k, i):
    return np.array(f[cd[k][i, 0] if cd[k].shape[1] == 1 else cd[k][0, i]])
for k in ("Bin10", "Bin1", "spikeTimes", "clusterID", "cellID", "channel", "depth", "CS_Bin1"):
    if k in cd:
        try:
            a = get(k, 0); print(k, "unit0 shape", a.shape, a.dtype, a.ravel()[:6])
        except Exception as e: print(k, "err", e)
b = get("Bin10", 0); print("Bin10 row0 first/last", b[0][:3], b[0][-3:], "dt", np.diff(b[0][:5]), "len", b.shape)
if "Bin1" in cd:
    b1 = get("Bin1", 0); print("Bin1 unit0", b1.shape, b1[0][:4], np.diff(b1[0][:4]))
st = get("spikeTimes", 0).ravel(); print("spikeTimes unit0 n", len(st), st[:5], st[-2:])
rs = f["ReachS"]; print("ReachS keys", list(rs.keys()))
fk = np.array(f[rs["filt_kin"][0, 0]]); print("filt_kin reach0", fk.shape, fk[:, :3])
