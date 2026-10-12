# E10 · Operating point: amendment checks and H2-H4 on real data

Status: see `experiment.json` · Owner: grad-researcher · Started: 2026-10-12

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> Pre-registration: [PREREGISTRATION_OPERATING_POINT.md](../09_claude_science_handoff/reports/PREREGISTRATION_OPERATING_POINT.md) (v3, amendments 1-4). Background: HANDOFF sections 8-9.

## Question
With J held fixed in time, does the influence of the couplings on population activity (multi-information per neuron, coupling covariance) change in the reach window relative to baseline, and is the change explained by neurons moving toward or away from p = 0.5? Bears on whether the paper can claim a mechanism for reach-related population structure beyond rate changes.

Note: E09 H1 already found 60% of neurons move away from 0.5 and session-mean p(1-p) falls in 17/21 sessions (p = 0.018), so fixed J predicts influence falls. The pre-registration counts this as refutation of the original direction; H2-H4 are still run as registered, and the likely replacement claim ("PCs polarize; population becomes more rate-driven") is only a candidate, not tested here.

## Hypothesis and decision rule
As pre-registered (sections 5-6, amendments 1-4). Supported only if H0 passes AND H2: dI/N > 0 (p_FDR < 0.05, above own shuffle null) AND H3 holds AND H4a rho > 0 (p_FDR < 0.05). Refuted if dI/N <= 0, dI unrelated to dv, H3a/H3b fail, or the effect is 2022-only. Primary family {H2, H3, H4a}. H5 is reported only as a readout of chi_J (amendment 4).

Step 1 (synthetic, no cluster) must pass before step 2:
- Amendment 2: kappa test, J_eff = (1+kappa)J in reach window. Scenarios A (kappa ~ 0) and C (J x 1.6, kappa ~ 0.6 recovered, detected).
- Amendment 3: H4b as rho_real - rho_null. Scenario D must give ~0; scenario E must give < 0.
Pass criteria and outcome are logged as a dated amendment (5) in the pre-registration. No changes to the real-data analysis after real-data results are seen.

## Data
- Step 1: synthetic only (labelled as method validation), built from real fitted J matrices (N = 11, 15; 20 and 60 reaches, T = 700, 5 replicates, as in H0).
- Step 2: 21 sessions / 63 conditions, 10-ms bins binarized (count > 0), rows 100:800, bin 400 = outward 1-cm crossing. Primary: `exclude == 1` reaches dropped (1092 reaches); sensitivity: all 1249. Input paths follow E08 (MCH refit, exclusions applied); copy inputs under `$HOME` because the container cannot see `/data001` from sbatch.

## Method
- Model: time-varying-field pairwise model, fixed J, cubic B-spline h_i(t) (`tvfield_model.bspline_basis(400)`), exact ML (N <= 20); for N > 20, 10 random 16-neuron subsets per condition (same subsets for real data and nulls; amendment 1), check vs full population in the four N = 17-19 sessions.
- Windows (bins rel. 400): baseline -100..-40, reach -20..+20 (primary), late +50..+150 (exploratory). Delta = reach mean - baseline mean.
- Per condition: fit, `thermo()`, dI/N, dchi_J/N, dv; 20 trial-shuffle refits (per-neuron reach order permuted); kappa fit with held-out gain; empirical time-locked covariance (mean removed, debiased for reach count) vs fixed-J prediction (5-fold sensitivity).
- Statistics: unit = session (n = 21, conditions averaged), two-sided Wilcoxon signed-rank, BH-FDR over {H2, H3, H4a}; mixed model with mouse (6) random effect; cohort 2021 vs 2022 reported separately; sensitivity N >= 5.
- Reuses `../09_claude_science_handoff/code/{thermo_model,tvfield_model,run_h0}.py` by path and git commit (record commit in RUN_LOG).

## How to reproduce
```bash
# from the repo root, on the cluster (never run heavy work on the login node)
bash energy_over_time_organized/10_operating_point_h2_h4/code/run.sh
```
Scripts in `code/` (to be written; none run yet):
- `code/run.sh`: entry point; step 1 (synthetic, small), then step 2 submitted via sbatch.
- `code/step1_synthetic_amendments.py`: scenarios A, C, D, E for kappa test and rho_real - rho_null (planned).
- `code/step2_real_h2_h4.py`: per-condition fits, thermo, shuffles, kappa, covariance (planned).
- `code/step3_stats.py`: session-level tests, mixed models, figures/tables (planned).

## Outputs
- `results/` (cluster only, gitignored): raw outputs.
- `assets/figures/`, `assets/tables/` (committed): every figure/table cited in `RESULTS.md`.
