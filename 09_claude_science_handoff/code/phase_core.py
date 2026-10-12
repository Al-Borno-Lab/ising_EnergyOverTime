
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
T = 700
SIGNALS = ["energy", "j", "h", "firing_rate"]
LABEL = {"energy": "Energy E", "j": "Coupling term (ΣJss)", "h": "Field term (Σhs)", "firing_rate": "Firing rate"}
COLOR = {"energy": "#333333", "j": "#7B3294", "h": "#E08214", "firing_rate": "#1B7837"}
PHASES = ["baseline", "acceleration", "deceleration", "post"]
BASE = (-100, -40); POST_LEN = 50

def build_arrays(D, LM):
    out = {}
    for (s, st), g in D.groupby(["session", "stim"]):
        g = g.sort_values(["reach_idx", "t"])
        arr = {c: g[c].values.reshape(-1, T) for c in SIGNALS + ["x"]}
        V = np.diff(arr["x"], axis=1, prepend=arr["x"][:, :1])
        Vs = savgol_filter(V, 11, 3, axis=1)
        arr["vel"] = Vs; arr["acc"] = np.gradient(Vs, axis=1)
        lm = LM[(LM.session == s) & (LM.stim == st)].iloc[0].to_dict()
        arr["lm"] = lm
        arr["n_neurons"] = int(g.n_neurons.iloc[0])
        out[(s, st)] = arr
    return out

def phase_slices(lm):
    zc = int(lm["zero_cross"])
    return {"baseline": slice(zc + BASE[0], zc + BASE[1]),
            "acceleration": slice(int(lm["onset"]), zc),
            "deceleration": slice(zc, int(lm["decel_end"])),
            "post": slice(int(lm["decel_end"]), int(lm["decel_end"]) + POST_LEN)}
