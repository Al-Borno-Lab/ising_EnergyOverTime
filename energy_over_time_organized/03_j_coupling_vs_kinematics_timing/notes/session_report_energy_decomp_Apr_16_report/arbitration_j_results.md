# J-Value Arbitration Results

*Generated: 2026-04-16*

**Signal compared:** J (coupling values) vs Firing Rate  
**Data folder:** `/data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_Apr_16/`  
**Window:** [390, 410]  
**Reference:** acceleration  
**Reps:** 1 – 1  
**Stimuli:** 0 – 2

---

## Overall Results (per Rep)

| Rep | J Wins | Firing Wins | Ties | N Sessions | J Win % |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 35 | 19 | 9 | 63 | 55.6% |

## Per-Stimulus Breakdown

| Rep | Stim | J Wins | Firing Wins | Ties | N Sessions | J Win % | Mean ΔJ | Mean ΔFiring |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 12 | 6 | 3 | 21 | 57.1% | 3.000 | 3.810 |
| 1 | 1 | 13 | 6 | 2 | 21 | 61.9% | 2.619 | 3.524 |
| 1 | 2 | 10 | 7 | 4 | 21 | 47.6% | 4.000 | 4.000 |

## Closest Extremum to Kinematic Reference

Counts: how many sessions had each extremum closest to the acceleration peak.

| Rep | Stim | J Min | J Max | Firing Min | Firing Max |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 8 | 7 | 4 | 2 |
| 1 | 1 | 9 | 6 | 4 | 2 |
| 1 | 2 | 9 | 5 | 6 | 1 |

## Correlation: J vs Firing Rate

| Rep | Stim | Mean Cov(J,FR) | Mean Corr(J,FR) |
|:---:|:---:|:---:|:---:|
| 1 | 0 | -0.0248 | 0.0007 |
| 1 | 1 | -0.0390 | -0.0148 |
| 1 | 2 | -0.0185 | 0.1457 |

## Correlation: J vs Kinematics

Does the J (coupling) time series co-vary with the movement signals?

| Rep | Stim | Mean Corr(J, Accel) | Mean Corr(J, Vel) |
|:---:|:---:|:---:|:---:|
| 1 | 0 | 0.0149 | 0.0222 |
| 1 | 1 | 0.0036 | -0.0542 |
| 1 | 2 | -0.0016 | 0.0061 |

## Timing Analysis: Does J Lead or Lag the Kinematic Reference?

Two complementary metrics:

- **Signed extremum lag** (`j_best_idx − ref_idx`): negative → J extremum *precedes* the kinematic peak (J leads), positive → J *follows* (J lags).
- **Cross-correlation lag** (whole time series): positive → J leads the reference signal, negative → J lags.  Critical for causal interpretation.

| Rep | Stim | Mean Signed Lag (bins) | Std | XCorr Lag (bins) | Interpretation |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 1.6 | 3.3 | 0.9 | J lags kinematics |
| 1 | 1 | -0.2 | 3.4 | -80.0 | J leads kinematics |
| 1 | 2 | 1.7 | 5.1 | -62.7 | J lags kinematics |

## Output Files

**Rep 1**  
- Results CSV: `/home/ceas/enserrog/ising_EnergyOverTime/energy_over_time_script/arbitration_J/1/results_j.csv`  
- Session summary: `/home/ceas/enserrog/ising_EnergyOverTime/energy_over_time_script/arbitration_J/1/session_summary_j.csv`  
- J-win plots: `/home/ceas/enserrog/ising_EnergyOverTime/energy_over_time_script/arbitration_J/1/j_wins/`  
- Firing-win plots: `/home/ceas/enserrog/ising_EnergyOverTime/energy_over_time_script/arbitration_J/1/firing_wins/`  
- Tie plots: `/home/ceas/enserrog/ising_EnergyOverTime/energy_over_time_script/arbitration_J/1/ties/`  
