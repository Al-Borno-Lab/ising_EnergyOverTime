# E15 · Synchronous pauses: shared drive vs fixed coupling

Status: see `experiment.json` · Owner: orchestrator · Started: 2026-10-27

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> The binding design is `PREREGISTRATION_2026-10-27.md`; this page summarizes it.

## Question
During the reach, the field term drops while the couplings J stay fixed (A8, E12, S2). When many Purkinje cells pause together, is that:
- (a) only shared, reach-locked drive (rates);
- (b) shared drive amplified by the **fixed** couplings;
- (c) something a fixed-J pairwise model cannot produce?

This matters for the paper because synchronous Purkinje-cell pauses are what the downstream nuclei read. The answer says what the fitted couplings *do*, beyond "M1 fits better than M0".

## Hypothesis and decision rule
Pause = K (cells spiking in a 10-ms bin) at or below the training-baseline 10th percentile. Compare the observed held-out event rate with the rates predicted by M0 (rates only) and M1 (rates + fixed J), in the reach window vs baseline:
- A_obs0: does the data's excess over rates-only grow during the reach?
- A_10: do the fixed couplings predict that growth?
- A_obs1: is anything left over beyond M1?

Outcomes (a), (b) and (c) are defined in the pre-registration (BH-FDR over the three, equivalence margin ±log 1.25).

## Data
- `09_claude_science_handoff/data/spikes_v2.npz` (cluster copy under `$HOME`): 63 session × stim conditions, 10-ms bins, 700 bins per reach. Reaches with `exclude_k == 1` are dropped, and the stored `fold_k` gives 5-fold CV.
- Landmarks: `09_claude_science_handoff/assets/tables/phase_landmarks.csv` (`zero_cross` = peak velocity).

## Method
- **Models:** `tvfield_model.py` M0 and M1, fitted per fold.
- **P(K | t):** Poisson-binomial for M0. For M1: exact enumeration for N ≤ 20, Gibbs sampling above (validated against exact).
- **Statistics:** unit = session (n = 21); Wilcoxon signed-rank; Hodges–Lehmann with bootstrap CI; BH-FDR; cohort reported.
- **Controls:** rest-control window (−2000 to −1600 ms), burst events, other quantiles.
- **Validation:** synthetic validation (S0–S2) on real-condition ground truths comes first.

## How to reproduce
```bash
bash 15_sync_pause/code/run.sh
```
- `code/e15_lib.py`: P(K) computation, events, windows.
- `code/synth_e15.py`: synthetic validation.
- `code/run_session.py`: one SLURM task per session.
- `code/stats_e15.py`: statistics and figures.

## Outputs
- Committed: `assets/tables/e15_conditions.csv`, `assets/tables/e15_tests.csv`, `assets/figures/e15_pause_timecourse.png`, plus the synthetic tables.
- Raw outputs: `results/` (gitignored).
