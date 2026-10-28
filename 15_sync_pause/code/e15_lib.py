"""E15 library: population-count distributions P(K | t) under M0 / M1, pause/burst events, pre-registered windows.
Spins s in {-1,+1}; P(s|t) ~ exp(sum_i h_i(t) s_i + 1/2 s.J.s) (tvfield_model convention, J symmetric, zero diagonal).
K = number of cells spiking (s = +1) in a 10-ms bin. Design: PREREGISTRATION_2026-10-27.md."""
import os, sys
import numpy as np
from scipy.special import expit, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "09_claude_science_handoff", "code"))
import tvfield_model as tv  # noqa: E402

# ---- pre-registered constants (asserted, and written into every output row) ----
WINDOWS = {"B": (-100, -41), "W": (-30, 10), "P": (11, 50), "C": (-200, -160)}   # inclusive bin offsets from zero_cross
TB = (-210, 60)                                                                   # analysed bins (offsets), covers all windows
EVENTS = {"pause": ("lo", 0.10), "burst": ("hi", 0.90), "pause05": ("lo", 0.05), "burst95": ("hi", 0.95)}
EXACT_MAX_N = 20
GIBBS = dict(chains=400, sweeps=250, burn=50)
NFOLD, NBASIS = 5, 67

def win_bins(zc, name):
    a, b = WINDOWS[name]; return np.arange(zc + a, zc + b + 1)

def tb_bins(zc):
    return np.arange(zc + TB[0], zc + TB[1] + 1)

def pk_indep(h):
    """Poisson-binomial P(K | t) for independent cells with p_i(t) = sigmoid(2 h_i(t)). h: (T, N) -> (T, N+1)."""
    T, N = h.shape; p = expit(2 * h); P = np.zeros((T, N + 1)); P[:, 0] = 1.0
    for i in range(N):
        P[:, 1:] = P[:, 1:] * (1 - p[:, i:i + 1]) + P[:, :-1] * p[:, i:i + 1]; P[:, 0] *= 1 - p[:, i]
    return P

def pk_exact(h, J, chunk=2 ** 14):
    """Exact P(K | t) by enumerating all 2^N states (states sorted by K so chunks hold few K values)."""
    T, N = h.shape; assert N <= EXACT_MAX_N
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8); del idx
    Kall = bits.sum(1); o = np.argsort(Kall, kind="stable"); bits, Kall = bits[o], Kall[o]
    logZK = np.full((T, N + 1), -np.inf)
    for a in range(0, 2 ** N, chunk):
        s = 2.0 * bits[a:a + chunk] - 1
        E = s @ h.T + 0.5 * np.einsum("si,ij,sj->s", s, J, s)[:, None]          # (chunk, T) log-weights
        Kc = Kall[a:a + chunk]
        for k in np.unique(Kc):
            logZK[:, k] = np.logaddexp(logZK[:, k], logsumexp(E[Kc == k], axis=0))
    return np.exp(logZK - logsumexp(logZK, axis=1, keepdims=True))

def gibbs(h, J, rng, chains, sweeps, burn, jscale=None):
    """Parallel single-site Gibbs chains, independent per bin. Returns (P(K|t) from sweeps >= burn, final states (T,C,N)).
    jscale: optional (T,) multiplier on J per bin (synthetic scenario S2)."""
    T, N = h.shape; js = np.ones(T) if jscale is None else np.asarray(jscale, float)
    S = np.where(rng.random((T, chains, N)) < expit(2 * h)[:, None, :], 1.0, -1.0)
    hist = np.zeros((T, N + 1)); ks = np.arange(N + 1)
    for sw in range(sweeps):
        for i in range(N):
            loc = h[:, None, i] + js[:, None] * (S @ J[:, i])
            S[:, :, i] = np.where(rng.random((T, chains)) < expit(2 * loc), 1.0, -1.0)
        if sw >= burn:
            hist += ((S > 0).sum(2)[:, :, None] == ks).sum(1)
    return hist / hist.sum(1, keepdims=True), S

def pk_model(h, J, rng):
    """P(K | t) under M1: exact for N <= 20, Gibbs above. Returns (P, method)."""
    if h.shape[1] <= EXACT_MAX_N: return pk_exact(h, J), "exact"
    return gibbs(h, J, rng, **GIBBS)[0], "gibbs"

def sample(h, J, R, rng, burn=300, jscale=None):
    """Synthetic data: R reaches x T bins, each bin an independent draw from the Ising model at that bin
    (Gibbs, R chains per bin, `burn` sweeps). Returns X (R, T, N) uint8."""
    if not np.any(J):
        return (rng.random((R,) + h.shape) < expit(2 * h)[None]).astype(np.uint8)
    _, S = gibbs(h, J, rng, chains=R, sweeps=burn + 1, burn=burn, jscale=jscale)
    return (S.transpose(1, 0, 2) > 0).astype(np.uint8)

def _threshold(Kb, kind, q):
    return int(np.quantile(Kb, q, method="lower" if kind == "lo" else "higher"))

def _evprob(P, kind, k):
    return P[:, :k + 1].sum(1) if kind == "lo" else P[:, k:].sum(1)

def analyse_condition(X, fold, zc, rng, fits=None):
    """Held-out event rates for one condition. X (R, 700, N) uint8, fold (R,) in 0..4, zc = zero_cross bin.
    Returns dict with time courses (over tb_bins) and per-fold thresholds. `fits` (optional) caches per-fold fits."""
    R, T, N = X.shape
    assert T == 700 and sorted(np.unique(fold)) == list(range(NFOLD)), (T, np.unique(fold))
    tb = tb_bins(zc); assert tb[0] >= 0 and tb[-1] < T
    for w in WINDOWS: assert np.all(np.isin(win_bins(zc, w), tb))
    B = tv.bspline_basis(zc); assert B.shape == (700, NBASIS), B.shape
    K = X.sum(2).astype(int)
    out = {ev: dict(o=np.zeros((R, len(tb))), q0=np.zeros((R, len(tb))), q1=np.zeros((R, len(tb))), k=[]) for ev in EVENTS}
    mK = dict(o=K[:, tb].astype(float), m0=np.zeros((R, len(tb))), m1=np.zeros((R, len(tb))))
    methods = set(); ks = np.arange(N + 1)
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        m0 = tv.fit(X[tr], B, "M0"); m1 = tv.fit(X[tr], B, "M1")
        P0 = pk_indep(m0["h"][tb]); P1, meth = pk_model(m1["h"][tb], m1["J"], rng); methods.add(meth)
        Kb = K[tr][:, win_bins(zc, "B")].ravel()
        for ev, (kind, q) in EVENTS.items():
            k = _threshold(Kb, kind, q); out[ev]["k"].append(k)
            out[ev]["o"][te] = (K[te][:, tb] <= k) if kind == "lo" else (K[te][:, tb] >= k)
            out[ev]["q0"][te] = _evprob(P0, kind, k)[None]; out[ev]["q1"][te] = _evprob(P1, kind, k)[None]
        mK["m0"][te] = (P0 @ ks)[None]; mK["m1"][te] = (P1 @ ks)[None]
    return dict(events=out, meanK=mK, tb=tb, zc=zc, R=R, N=N, method="+".join(sorted(methods)))

def window_stats(res):
    """Per-event, per-test-window statistics (pre-registered definitions). Returns list of dict rows."""
    zc, tb, R = res["zc"], res["tb"], res["R"]
    col = {w: np.isin(tb, win_bins(zc, w)) for w in WINDOWS}
    rows = []
    for ev, d in res["events"].items():
        rate = {}
        for w, c in col.items():
            eps = 0.5 / (R * c.sum())
            rate[w] = {s: d[s][:, c].mean() + eps for s in ("o", "q0", "q1")}
        for X in ("W", "P", "C"):
            b, x = rate["B"], rate[X]; L = np.log
            rows.append(dict(event=ev, test_window=X,
                R_obs=L(x["o"] / b["o"]),
                A_obs0=L(x["o"] / x["q0"]) - L(b["o"] / b["q0"]),
                A_10=L(x["q1"] / x["q0"]) - L(b["q1"] / b["q0"]),
                A_obs1=L(x["o"] / x["q1"]) - L(b["o"] / b["q1"]),
                L_obs0=L(b["o"] / b["q0"]), L_obs1=L(b["o"] / b["q1"]),
                e_obs_B=b["o"], e0_B=b["q0"], e1_B=b["q1"], e_obs_X=x["o"], e0_X=x["q0"], e1_X=x["q1"],
                meanK_obs_X=res["meanK"]["o"][:, col[X]].mean(), meanK_m0_X=res["meanK"]["m0"][:, col[X]].mean(),
                meanK_m1_X=res["meanK"]["m1"][:, col[X]].mean(), meanK_obs_B=res["meanK"]["o"][:, col["B"]].mean(),
                meanK_m0_B=res["meanK"]["m0"][:, col["B"]].mean(), meanK_m1_B=res["meanK"]["m1"][:, col["B"]].mean(),
                k_per_fold=";".join(map(str, d["k"])), N=res["N"], R=R, zc=zc, method=res["method"],
                windows=str(WINDOWS), quantile=EVENTS[ev][1]))
    return rows

def timecourses(res):
    """Arrays for figures: offsets from zc, and per event e_obs/e0/e1 (len(tb),)."""
    d = {"offset": res["tb"] - res["zc"]}
    for ev, v in res["events"].items():
        for s in ("o", "q0", "q1"): d[f"{ev}__{s}"] = v[s].mean(0)
    for s in ("o", "m0", "m1"): d[f"meanK__{s}"] = res["meanK"][s].mean(0)
    return d
