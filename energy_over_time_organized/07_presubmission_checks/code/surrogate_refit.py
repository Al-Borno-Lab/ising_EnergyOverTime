
from ising_common import *
from multiprocessing import Pool
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
NREAL = 3
dirs = cond_dirs()
if len(sys.argv) > 2: dirs = [d for d in dirs if tag(d)[0] in sys.argv[2].split(",")]
def task(args):
    d, kind, k = args
    s, st = tag(d); _, X, _ = load_spikes(d); N = X.shape[1]
    rng = np.random.default_rng(1000 * k + st)
    if kind == "surrogate":
        X = np.column_stack([rng.permutation(X[:, i]) for i in range(N)])
    t = time.time(); m = quiet_fit(X, seed=k + 1)
    J = m[N:]
    S = 2 * X - 1; C = np.corrcoef(S.T)[np.triu_indices(N, 1)]
    return dict(session=s, stim=st, N=N, kind=kind, k=k, mean_absJ=np.abs(J).mean(), rms_J=np.sqrt((J**2).mean()), sum_J=J.sum(),
                max_absJ=np.abs(J).max(), mean_abs_corr=np.nanmean(np.abs(C)), fit_s=time.time() - t)
tasks = [(d, "real", k) for d in dirs for k in range(NREAL)] + [(d, "surrogate", k) for d in dirs for k in range(NS)]
# big-N first for load balance
tasks.sort(key=lambda a: -pd.read_csv(os.path.join(a[0], "J_parameters.csv"), nrows=0).shape[1])
out = []
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
    for i, r in enumerate(p.imap_unordered(task, tasks, chunksize=1)):
        out.append(r)
        if i % 200 == 0:
            pd.DataFrame(out).to_csv("surrogate_refit_raw.csv", index=False); print(i, len(tasks), flush=True)
pd.DataFrame(out).to_csv("surrogate_refit_raw.csv", index=False)
