
from ising_common import *
from multiprocessing import Pool
from numba import njit
K = 5
@njit(cache=True)
def pair_term(S, J, N):
    T = S.shape[0]; out = np.zeros(T)
    for t in range(T):
        acc = 0.0
        for i in range(N - 1):
            for j in range(i + 1, N):
                acc += J[i, j] * S[t, i] * S[t, j]
        out[t] = acc
    return out
@njit(cache=True)
def metro_mean_pair(h, J, beta, n_sweeps, burn, seed):
    np.random.seed(seed); N = h.shape[0]; s = np.where(np.random.random(N) < 0.5, -1.0, 1.0); acc = 0.0; c = 0
    for it in range(burn + n_sweeps):
        for _ in range(N):
            i = np.random.randint(N); loc = h[i]
            for j in range(N): loc += beta * J[i, j] * s[j]
            dE = 2.0 * s[i] * loc
            if dE <= 0 or np.random.random() < np.exp(-dE): s[i] = -s[i]
        if it >= burn:
            e = 0.0
            for i in range(N - 1):
                for j in range(i + 1, N): e += J[i, j] * s[i] * s[j]
            acc += e; c += 1
    return acc / c
def unpack(m, N):
    h = m[:N]; J = np.zeros((N, N)); iu = np.triu_indices(N, 1); J[iu] = m[N:]; J = J + J.T; return h, J
def logZ_exact(h, J):
    N = len(h); tot = []
    for chunk in range(0, 2**N, 2**20):
        idx = np.arange(chunk, min(chunk + 2**20, 2**N))
        S = 2.0 * ((idx[:, None] >> np.arange(N)) & 1) - 1
        negE = S @ h + 0.5 * np.einsum("ti,ij,tj->t", S, J, S)
        tot.append(negE)
    a = np.concatenate(tot); mx = a.max(); return mx + np.log(np.exp(a - mx).sum())
def logZ_TI(h, J, nb=41, sweeps=4000, burn=1000):
    betas = np.linspace(0, 1, nb)
    vals = np.array([np.mean([metro_mean_pair(h, J, b, sweeps, burn, 17 * q + int(b * 1000)) for q in range(3)]) for b in betas])
    return np.sum(np.log(2 * np.cosh(h))) + (np.trapezoid if hasattr(np,'trapezoid') else np.trapz)(vals, betas)
def task(d):
    s, st = tag(d); df, X, reach = load_spikes(d); N = X.shape[1]; S = 2.0 * X - 1
    ureach = np.unique(reach); rng = np.random.default_rng(st); folds = np.array_split(rng.permutation(ureach), K)
    llr = np.full(len(X), np.nan); ll_i = np.full(len(X), np.nan); ll_n = np.full(len(X), np.nan); zinfo = []
    for f, test in enumerate(folds):
        te = np.isin(reach, test); tr = ~te
        m = quiet_fit(X[tr], seed=f + 1); h, J = unpack(m, N)
        lz = logZ_exact(h, J) if N <= 22 else logZ_TI(h, J)
        if N <= 22 and f == 0: zinfo.append(("check_TI_minus_exact", logZ_TI(h, J) - lz))
        p = np.clip(X[tr].mean(0), 1e-4, 1 - 1e-4)
        li = S[te] @ h + pair_term(S[te], J, N) - lz
        ln = (X[te] * np.log(p) + (1 - X[te]) * np.log(1 - p)).sum(1)
        ll_i[te] = li; ll_n[te] = ln; llr[te] = li - ln
    out = pd.DataFrame(dict(session=s, stim=st, reach_idx=reach, t=df.groupby("reach_idx").cumcount().values,
                            ll_ising=ll_i, ll_indep=ll_n, llr=llr, firing_rate=df.firing_rate.values, x=df.x.values, n_neurons=N))
    return out, dict(session=s, stim=st, N=N, **{k: v for k, v in zinfo})
dirs = cond_dirs()
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    res = p.map(task, sorted(dirs, key=lambda d: -pd.read_csv(os.path.join(d, "J_parameters.csv"), nrows=0).shape[1]), chunksize=1)
pd.concat([r[0] for r in res]).to_csv("timeresolved_llr.csv.gz", index=False, compression="gzip")
pd.DataFrame([r[1] for r in res]).to_csv("logZ_checks.csv", index=False)
