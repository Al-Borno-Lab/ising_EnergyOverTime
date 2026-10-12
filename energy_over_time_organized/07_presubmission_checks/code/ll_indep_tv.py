
from ising_common import *
from multiprocessing import Pool
from scipy.ndimage import uniform_filter1d
K = 5; T = 700
def task(d):
    s, st = tag(d); df, X, reach = load_spikes(d); N = X.shape[1]
    ureach = np.unique(reach); rng = np.random.default_rng(st); folds = np.array_split(rng.permutation(ureach), K)
    Xr = X.reshape(-1, T, N); ridx = reach.reshape(-1, T)[:, 0]
    out = np.full(Xr.shape[:2], np.nan)
    for sm in (1, 11):
        ll = np.full(Xr.shape[:2], np.nan)
        for test in folds:
            te = np.isin(ridx, test)
            p = Xr[~te].mean(0)                                  # T x N time-varying rates from training reaches
            if sm > 1: p = uniform_filter1d(p, sm, axis=0)
            nt = (~te).sum(); p = (p * nt + 0.5) / (nt + 1)      # add-half smoothing to avoid log 0
            ll[te] = (Xr[te] * np.log(p) + (1 - Xr[te]) * np.log(1 - p)).sum(2)
        if sm == 1: out1 = ll
        else: out11 = ll
    return pd.DataFrame(dict(session=s, stim=st, reach_idx=reach, t=np.tile(np.arange(T), Xr.shape[0]), ll_indep_tv=out1.ravel(), ll_indep_tv_s11=out11.ravel()))
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    res = p.map(task, cond_dirs())
pd.concat(res).to_csv("ll_indep_timevarying.csv.gz", index=False, compression="gzip")
