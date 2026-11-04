# E16 · 1-ms field-term peak and trough timing vs peak acceleration and peak velocity

Status: see `experiment.json` · Owner: orchestrator · Started: 2026-11-03

> This file is the **experiment write-up**: what we ask and how. The binding design is `PREREGISTRATION_2026-11-03.md`. Results go in `RESULTS.md`.

## Question
Does the 1-ms field term peak at peak acceleration and trough at peak velocity, condition by condition, when the spline basis is equally flexible everywhere? The exploratory median curve suggested this, but per-session checks with the standard basis did not.

## Hypothesis and decision rule
- **Hit:** the peak lies within ±30 ms of the condition's own peak acceleration; the trough lies within ±30 ms of peak velocity. Search window pv −200…+100 ms.
- **Test:** hit fraction in the reach window vs rest windows (shifted −2000 and −2500 ms), Wilcoxon over 21 sessions, BH-FDR over the two landmarks.
- **Bases:** primary is a uniform 50-ms-knot basis (U0). The claim is supported only if it replicates with knots offset by 25 ms (U25).

## Data
- **Spikes:** 1-ms spikes from the E14 extraction (`14_binsize_1ms_synchrony/results/spk/`; gates G1–G3 pass 21/21). Path passed as an argument.
- **Conditions:** 63, with exclusions.
- **Landmarks:** `09_claude_science_handoff/assets/tables/phase_landmarks.csv`.

## Method
- **Model:** M1 by pseudo-likelihood at 1 ms, with three bases (U0, U25 and STD).
- **Signal:** the field term per condition.
- **Validation:** synthetic Z0 (null) and Z1 (planted peak and trough) come first.

## How to reproduce
```bash
bash 16_onems_timing/code/run.sh submit
bash 16_onems_timing/code/run.sh stats
```

## Outputs
`assets/tables/e16_*.csv|json` and `assets/figures/e16_onems_timing.png`. Raw outputs go in `results/` (gitignored).
