import sys, json, numpy as np
try:
    import h5py
except ImportError:
    print("NO_H5PY"); sys.exit(3)
D = "/data001/projects/enserrog/dylan/AbigailUpdated"
files = sys.argv[1:]
def desc(f, obj, depth=0, maxref=3):
    if isinstance(obj, h5py.Group):
        return {k: desc(f, obj[k], depth + 1) for k in list(obj.keys())[:60]} if depth < 4 else f"group({len(obj)})"
    d = {"shape": obj.shape, "dtype": str(obj.dtype), "mclass": obj.attrs.get("MATLAB_class", b"").decode() if isinstance(obj.attrs.get("MATLAB_class", b""), bytes) else str(obj.attrs.get("MATLAB_class"))}
    if obj.dtype == h5py.ref_dtype and depth < 4:
        refs = obj[()].ravel()[:maxref]
        d["first_refs"] = [desc(f, f[r], depth + 1) for r in refs]
    elif obj.size <= 40 and obj.dtype != h5py.ref_dtype:
        v = obj[()]
        d["value"] = "".join(map(chr, v.ravel())) if d["mclass"] == "char" else np.asarray(v).ravel().tolist()
    return d
def mstr(f, ref):
    o = f[ref]; v = o[()]
    if o.attrs.get("MATLAB_class", b"") == b"char": return "".join(map(chr, v.ravel()))
    return None if o.attrs.get("MATLAB_empty", 0) else np.asarray(v).ravel().tolist()[:5]
out = {}
for fn in files:
    with h5py.File(f"{D}/{fn}", "r") as f:
        rec = {"top": list(f.keys())}
        for top in ["cellData", "ReachS"]:
            if top in f:
                g = f[top]; rec[top] = {}
                for k in g.keys():
                    ds = g[k]
                    if ds.dtype == h5py.ref_dtype:
                        refs = ds[()].ravel()
                        first = f[refs[0]]
                        rec[top][k] = {"n": len(refs), "first_shape": first.shape, "first_class": str(first.attrs.get("MATLAB_class")),
                                       "first_value": mstr(f, refs[0]) if first.size < 60 else None}
                        if k in ("cellID", "stim", "exclude"):
                            rec[top][k]["values"] = [str(mstr(f, r)) for r in refs]
                    else:
                        rec[top][k] = {"shape": ds.shape, "dtype": str(ds.dtype)}
        for top in rec["top"]:
            if top not in ("cellData", "ReachS", "#refs#"):
                try: rec[f"other_{top}"] = desc(f, f[top], 2)
                except Exception as e: rec[f"other_{top}"] = str(e)
        out[fn] = rec
json.dump(out, open("src_structure.json", "w"), indent=1, default=str)
print(json.dumps(out, default=str)[:6000])
