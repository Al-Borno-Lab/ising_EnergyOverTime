"""E15 library check: exact P(K) vs brute force; Gibbs vs exact on real M1 fits (fold 0 training data).
    python validate_lib.py SPIKES_NPZ LANDMARKS_CSV OUT_CSV"""
import sys, itertools
import numpy as np, pandas as pd
import e15_lib as L
tv = L.tv
Z = dict(np.load(sys.argv[1])); LM = pd.read_csv(sys.argv[2]); rng = np.random.default_rng(15); rows = []
N, T = 5, 3; h = rng.normal(0, .5, (T, N)); J = rng.normal(0, .3, (N, N)); J = (J + J.T) / 2; np.fill_diagonal(J, 0)
S = np.array(list(itertools.product([-1, 1], repeat=N)), float); P = np.zeros((T, N + 1))
for t in range(T):
    w = np.exp(S @ h[t] + .5 * np.einsum("si,ij,sj->s", S, J, S)); w /= w.sum()
    for s, p in zip(S, w): P[t, int((s > 0).sum())] += p
rows.append(dict(check="exact_vs_bruteforce_N5", max_abs_diff=np.abs(L.pk_exact(h, J) - P).max()))
rows.append(dict(check="exact_J0_vs_poissonbinomial", max_abs_diff=np.abs(L.pk_exact(h, 0 * J) - L.pk_indep(h)).max()))
for k in ("210421_0", "210515_0", "220516_0"):
    s, st = map(int, k.split("_")); m = Z["exclude_" + k] == 0; X = Z["X_" + k][m].astype(np.uint8); f = Z["fold_" + k][m]
    zc = int(LM[(LM.session == s) & (LM.stim == st)].zero_cross.iloc[0])
    m1 = tv.fit(X[f != 0], tv.bspline_basis(zc), "M1"); hb = m1["h"][L.tb_bins(zc)]
    Pe = L.pk_exact(hb, m1["J"]); Pg = L.gibbs(hb, m1["J"], rng, **L.GIBBS)[0]
    Kb = X[f != 0].sum(2)[:, L.win_bins(zc, "B")].ravel(); klo = int(np.quantile(Kb, .1, method="lower"))
    rows.append(dict(check=f"gibbs_vs_exact_{k}_N{X.shape[2]}", max_abs_diff=np.abs(Pe - Pg).max(),
                     pause_prob_exact=Pe[:, :klo + 1].sum(1).mean(), pause_prob_gibbs=Pg[:, :klo + 1].sum(1).mean()))
pd.DataFrame(rows).to_csv(sys.argv[3], index=False); print(pd.DataFrame(rows).to_string(index=False))
