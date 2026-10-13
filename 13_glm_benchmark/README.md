# E13 · Coupled-GLM benchmark against the time-varying-field Ising model

Status: see `experiment.json` · Owner: unassigned · Started: 2026-10-12

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> Motivation: `docs/literature/METHODS_COMPARISON_REPORT.md` §2.3 and §5 (PR #5). The likely reviewer objection is
> "a GLM could do this." We found no coupled-GLM analysis of simultaneously recorded Purkinje cells.

## Question
On the same reaches and folds, how does a standard coupled GLM compare with our time-varying-field Ising model (M1)?
In particular:
- (a) Does coupling beyond reach-locked rates (A7) appear in a GLM, and is it same-bin (synchronous) or lagged?
- (b) Does a GLM also show no change in coupling during the reach (A8)?
- (c) What does the Ising model give that the GLM can't: a normalized joint likelihood, P(K), and thermodynamic quantities?

The point is to state honestly which of our results are method-independent (stronger) and which need the Ising model (our contribution).

## Hypotheses (freeze in a dated pre-registration before fitting real data)
- **G1:** A GLM with a spline PSTH baseline and **same-bin** coupling matches M1's held-out conditional log-likelihood (it is mechanically the same model; this is a sanity check of the implementation).
- **G2:** A GLM with **1-bin-lag (10 ms)** coupling only recovers less than half of M1's coupling gain over the rates-only model (M0), as Mlag did at 200 ms (about a quarter).
- **G3:** GLM coupling weights don't change between the baseline and reach epochs (epoch interaction term; same windows as A8). Report the CI, not just p.
- **G4 (descriptive):** Simulate population patterns from each fitted model and compare P(K) with the data. Only the Ising model is expected to reproduce P(K) without extra terms.

Statistics: unit = session (n = 21), two-sided Wilcoxon, BH-FDR over {G2, G3}; cohort reported separately; mouse as a random effect.

## Data
`spikes_v2.npz` (cluster, `09_claude_science_handoff/data/`), with `exclude_k == 0`. 10-ms bins, rows 100:800, the same
5-fold reach split as E07 (`fold_k`), and phase landmarks from `09_.../assets/tables/phase_landmarks.csv`.

## Method
- Models, each fitted per neuron with logistic regression:
  - **GLM-0:** spline PSTH baseline only. This is M0 and serves as the reference.
  - **GLM-sync:** baseline plus same-bin coupling to the other cells.
  - **GLM-lag1:** baseline plus coupling to the other cells' previous bin, plus spike history.
  - **GLM-lag1+hist:** lagged coupling plus a short self-history filter (30 ms).
  - Also report **GLM-sync+lag1**.
- Spline basis identical to `tvfield_model.bspline_basis(400)`. L2 penalty chosen by nested CV.
- Metric: held-out conditional log-likelihood per bin (nats/bin), by epoch (baseline, acceleration, deceleration, post).
- G3: an epoch × coupling interaction in GLM-sync, refit with 20 trial shuffles as the null (same as A7).
- Compare against stored M0/M1 CV scores (`07_.../timevarying_field/tvfield_cv_scores.csv`). Refit M1 with exclusions if needed.
- Compute is light (logistic regressions). Run as a single sbatch job with 16–32 CPUs.

## How to reproduce
```bash
bash 13_glm_benchmark/code/run.sh
```
Planned scripts: `code/fit_glms.py`, `code/simulate_pk.py`, `code/stats_e13.py`.

## Outputs
`assets/tables/glm_cv_scores.csv`, `assets/figures/glm_vs_ising.png`; raw outputs go in `results/` (gitignored).
