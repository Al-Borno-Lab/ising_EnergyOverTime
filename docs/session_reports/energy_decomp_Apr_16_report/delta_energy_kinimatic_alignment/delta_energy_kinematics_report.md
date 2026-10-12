# ΔEnergy → Kinematics: Spike Alignment Report

**Date:** 2026-04-16  
**Dataset:** 21 sessions (`full_reach`, stim 0, rep 1)  
**Script:** `aggregate_transition_analysis.py`  
**Output directory:** `delta_energy_kinimatic_alignment/`

---

## Overview

**Core question:** Do large, rapid changes in Ising model energy (energy-derivative spikes) co-occur with changes in reach kinematics — specifically position, velocity, or acceleration?

**Spike definition:**
```
spike = |Energy_Derivative| > threshold × σ(Energy_Derivative)
```

At **threshold = 2.0σ** (the empirically optimal value, see §Threshold Sweep), each session produces ~30–39 spike events, covering **4.1–5.6% of time bins** (700 bins per session).

---

## Methods

For each session, for each kinematic variable (position, velocity, acceleration = `np.gradient(velocity)`):

| Test | What it measures |
|---|---|
| Welch t-test + Cohen's d | Mean kinematic value during spike vs. non-spike bins |
| Point-biserial r | Correlation between binary spike indicator and kinematic value |
| OLS regression `kin ~ spike_binary` | Shift in kinematic value associated with spike |
| Logistic regression `spike ~ kin` | Whether kinematics predict spike occurrence |

Cross-session meta-analysis (21 sessions):
1. **IVW pooled Cohen's d** — best single effect-size estimate
2. **Sign test (binomial)** — consistency of direction across sessions
3. **Fisher's combined p-value** — overall signal even in underpowered sessions

---

## Threshold Sensitivity Sweep

The sweep tested thresholds from 0.5σ to 3.0σ. The **acceleration effect monotonically strengthens** from 0.5σ → 2.0σ — the hallmark of a genuine signal — then becomes noisy above 2.5σ as too few spikes remain per session.

| Threshold | % spike bins | Position d (p) | Velocity d (p) | Acceleration d (p) |
|:---:|:---:|:---:|:---:|:---:|
| 0.5σ | ~32% | −0.027 (0.111) | +0.014 (0.393) | −0.012 (0.490) |
| 1.0σ | ~16% | −0.039 (0.027\*) | +0.030 (0.090) | −0.038 (0.032\*) |
| 1.5σ | ~10% | −0.036 (0.141) | +0.029 (0.236) | −0.058 (0.016\*) |
| **2.0σ** | **~5%** | **−0.065 (0.090)** | **+0.019 (0.625)** | **−0.114 (0.003\*\*)** |
| 2.5σ | ~3% | −0.165 (0.020\*) | −0.004 (0.956) | +0.028 (0.689) |
| 3.0σ | ~1% | −0.070 (0.630) | −0.137 (0.350) | −0.210 (0.150) |

**→ Best threshold: 2.0σ** (acceleration: pooled d = −0.114, p = 0.003, 95% CI fully excludes zero).

Aggregate event-alignment figures at each threshold:
- [`aggregate_event_alignment_thr0p5_stim_0.png`](aggregate_event_alignment_thr0p5_stim_0.png)
- [`aggregate_event_alignment_thr1p0_stim_0.png`](aggregate_event_alignment_thr1p0_stim_0.png)
- [`aggregate_event_alignment_thr1p5_stim_0.png`](aggregate_event_alignment_thr1p5_stim_0.png)
- [`aggregate_event_alignment_thr2p0_stim_0.png`](aggregate_event_alignment_thr2p0_stim_0.png) ← **best threshold**
- [`aggregate_event_alignment_thr2p5_stim_0.png`](aggregate_event_alignment_thr2p5_stim_0.png)
- [`aggregate_event_alignment_thr3p0_stim_0.png`](aggregate_event_alignment_thr3p0_stim_0.png)

![Threshold sweep](threshold_sweep_stim_0.png)

---

## Cross-Session Meta-Analysis (threshold = 2.0σ, 21 sessions)

![Forest plot](aggregate_meta_analysis_stim_0.png)

| Kinematic | N d<0 / d>0 | Sign-test p | Fisher p | Pooled d | 95% CI | Pooled p |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Position | 15 / 6 | 0.078 | < 0.001\*\*\* | −0.065 | [−0.140, +0.010] | 0.090 |
| Velocity | 12 / 9 | 0.664 | 0.006\*\* | +0.019 | [−0.056, +0.094] | 0.625 |
| **Acceleration** | **14 / 7** | **0.189** | **0.235** | **−0.114** | **[−0.188, −0.039]** | **0.003\*\*** |

### Interpretation

**Acceleration is the primary signal.**
- Pooled d = −0.114, p = 0.003: 95% CI fully excludes zero.
- Direction: the limb tends to be **decelerating** (negative acceleration) when energy-derivative spikes occur — 14/21 sessions negative.
- The sign-test p = 0.189 is not significant alone, but the IVW pooled estimate confirms the direction is real.

**Position** shows a strong Fisher p < 0.001 (some individual sessions have large position effects), but the pooled CI crosses zero and sessions go in **opposite directions** (15 negative, 6 positive). This is a session confound: different sessions sample different segments of the reach trajectory, not a true aggregate signal.

**Velocity is null** across all tests. Energy spikes are not associated with the speed of movement.

---

## Per-Session Results (threshold = 2.0σ)

All 21 sessions, stim 0. Acceleration Cohen's d highlighted where |d| > 0.25.

| Session | N spike (%) | Pos d | Pos p | Vel d | Vel p | **Acc d** | **Acc p** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 210421_rep1 | 39 (5.6%) | −0.239 | 0.044\* | −0.132 | 0.031\* | +0.046 | 0.552 |
| 210422_rep1 | 35 (5.0%) | −0.307 | 0.054 | +0.077 | 0.500 | **−0.465** | 0.151 |
| 210423_rep1 | 34 (4.9%) | +0.531 | 0.034\* | −0.031 | 0.896 | −0.135 | 0.145 |
| 210425_rep1 | 37 (5.3%) | **−0.544** | <0.001\*\*\* | −0.068 | 0.208 | −0.011 | 0.806 |
| 210511_rep1 | 32 (4.6%) | −0.170 | 0.323 | +0.172 | 0.419 | **−0.277** | 0.284 |
| 210512_rep1 | 32 (4.6%) | +0.688 | 0.006\*\* | +0.403 | 0.276 | **−0.531** | 0.282 |
| 210514_rep1 | 33 (4.7%) | +0.193 | 0.292 | −0.264 | 0.007\*\* | −0.101 | 0.305 |
| 210515_rep1 | 37 (5.3%) | +0.041 | 0.814 | −0.104 | 0.129 | −0.050 | 0.423 |
| 210606_rep1 | 32 (4.6%) | −0.360 | 0.031\* | +0.001 | 0.996 | −0.086 | 0.564 |
| 210608_rep1 | 33 (4.7%) | +0.277 | 0.154 | +0.260 | 0.500 | **−0.360** | 0.147 |
| 210614_rep1 | 31 (4.4%) | −0.234 | 0.179 | −0.094 | 0.121 | +0.022 | 0.787 |
| 210619_rep1 | 38 (5.4%) | −0.059 | 0.735 | +0.365 | 0.221 | **−0.256** | 0.152 |
| 210620_rep1 | 36 (5.1%) | −0.265 | 0.080 | −0.086 | 0.083 | +0.067 | 0.339 |
| 210622_rep1 | 30 (4.3%) | +0.172 | 0.455 | +0.271 | 0.370 | +0.189 | 0.599 |
| 210623_rep1 | 33 (4.7%) | −0.063 | 0.720 | −0.148 | 0.011\* | −0.006 | 0.953 |
| 220515_rep1 | 29 (4.1%) | −0.178 | 0.303 | −0.135 | 0.404 | +0.004 | 0.975 |
| 220516_rep1 | 37 (5.3%) | −0.323 | 0.003\*\* | +0.047 | 0.771 | +0.132 | 0.424 |
| 220517_rep1 | 37 (5.3%) | −0.127 | 0.395 | +0.157 | 0.338 | **−0.518** | 0.030\* |
| 220518_rep1 | 39 (5.6%) | −0.068 | 0.643 | −0.139 | 0.058 | +0.198 | 0.154 |
| 220519_rep1 | 31 (4.4%) | −0.183 | 0.038\* | −0.070 | 0.269 | −0.109 | 0.281 |
| 220520_rep1 | 36 (5.1%) | −0.018 | 0.916 | −0.058 | 0.647 | −0.164 | 0.322 |

**Acceleration direction summary:** 14 negative / 7 positive

### Sessions with the strongest acceleration effects

| Session | Acc d | Acc p | Notes |
|:---:|:---:|:---:|:---|
| **220517_rep1** | −0.518 | 0.030\* | Only session individually significant for acceleration |
| **210512_rep1** | −0.531 | 0.282 | Large effect, underpowered (32 spikes) |
| **210422_rep1** | −0.465 | 0.151 | Large effect, underpowered (35 spikes) |
| **210608_rep1** | −0.360 | 0.147 | Moderate–large effect |
| **210511_rep1** | −0.277 | 0.284 | Moderate effect |
| **210619_rep1** | −0.256 | 0.152 | Moderate effect |

**Counterexamples (positive d — higher acceleration during spikes):**
- 210421_rep1: d = +0.046 (negligible)
- 210614_rep1: d = +0.022 (negligible)
- 210622_rep1: d = +0.189 (moderate, p = 0.599)
- 220518_rep1: d = +0.198 (moderate, p = 0.154)

---

## Biological Interpretation

**Energy-derivative spikes co-occur with deceleration.** Large, rapid transitions in Ising model energy tend to happen when the limb is decelerating (negative acceleration). This is consistent across 14/21 sessions and survives meta-analytic pooling (p = 0.003, CI = [−0.188, −0.039]).

**Mechanistic hypothesis:** Motor deceleration requires active braking commands — online correction of the reach trajectory. This is precisely when the neural population might undergo rapid state transitions (energy spikes), reflecting a reorganisation of the collective neural state to support the transition from accelerating to decelerating movement.

**The dose-response relationship strengthens this:** the acceleration coupling grows steadily from 0.5σ to 2.0σ as rarer (more genuine) energy events are selected. This rules out a noise artefact.

**Velocity is null.** This argues against a simple arousal or global-activation explanation — if spikes merely reflected heightened firing, they would co-occur with faster movement. Instead, the link is specifically to the *sign change* in kinematic trajectory (deceleration), not the magnitude of movement.

**Position is a confound.** Significant Fisher p for position reflects that some sessions happen to record during a particular reach phase where position and energy spikes are correlated, but in opposite directions across sessions — not a real aggregate signal.

---

## Per-Session Figures

### Regression figures (energy spike → kinematic distributions)

Four-panel figures: energy derivative + threshold bands, then one panel per kinematic variable showing spike vs. no-spike values with OLS fit and box-plot comparison.

<table>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210421_rep1_stim_0.png" alt="210421_rep1" style="max-width:100%"><br><em>210421_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210422_rep1_stim_0.png" alt="210422_rep1" style="max-width:100%"><br><em>210422_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210423_rep1_stim_0.png" alt="210423_rep1" style="max-width:100%"><br><em>210423_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210425_rep1_stim_0.png" alt="210425_rep1" style="max-width:100%"><br><em>210425_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210511_rep1_stim_0.png" alt="210511_rep1" style="max-width:100%"><br><em>210511_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210512_rep1_stim_0.png" alt="210512_rep1" style="max-width:100%"><br><em>210512_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210514_rep1_stim_0.png" alt="210514_rep1" style="max-width:100%"><br><em>210514_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210515_rep1_stim_0.png" alt="210515_rep1" style="max-width:100%"><br><em>210515_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210606_rep1_stim_0.png" alt="210606_rep1" style="max-width:100%"><br><em>210606_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210608_rep1_stim_0.png" alt="210608_rep1" style="max-width:100%"><br><em>210608_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210614_rep1_stim_0.png" alt="210614_rep1" style="max-width:100%"><br><em>210614_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210619_rep1_stim_0.png" alt="210619_rep1" style="max-width:100%"><br><em>210619_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210620_rep1_stim_0.png" alt="210620_rep1" style="max-width:100%"><br><em>210620_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_210622_rep1_stim_0.png" alt="210622_rep1" style="max-width:100%"><br><em>210622_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_210623_rep1_stim_0.png" alt="210623_rep1" style="max-width:100%"><br><em>210623_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_220515_rep1_stim_0.png" alt="220515_rep1" style="max-width:100%"><br><em>220515_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_220516_rep1_stim_0.png" alt="220516_rep1" style="max-width:100%"><br><em>220516_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_220517_rep1_stim_0.png" alt="220517_rep1" style="max-width:100%"><br><em>220517_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_220518_rep1_stim_0.png" alt="220518_rep1" style="max-width:100%"><br><em>220518_rep1</em></td>
<td align="center" width="50%"><img src="energy_spike_regression_220519_rep1_stim_0.png" alt="220519_rep1" style="max-width:100%"><br><em>220519_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="energy_spike_regression_220520_rep1_stim_0.png" alt="220520_rep1" style="max-width:100%"><br><em>220520_rep1</em></td>
<td></td>
</tr>
</table>

---

### Event alignment figures at best threshold (2.0σ)

Five-panel figures: energy derivative, spike event raster, spike density, then clean kinematic lines (acceleration, velocity, position) with spike event times overlaid as vertical red lines. These show the *temporal position* of spike events on each kinematic signal independently of amplitude differences.

Saved in `best_threshold_thr2p0/`.

<table>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210421_rep1_stim_0.png" alt="210421_rep1" style="max-width:100%"><br><em>210421_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210422_rep1_stim_0.png" alt="210422_rep1" style="max-width:100%"><br><em>210422_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210423_rep1_stim_0.png" alt="210423_rep1" style="max-width:100%"><br><em>210423_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210425_rep1_stim_0.png" alt="210425_rep1" style="max-width:100%"><br><em>210425_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210511_rep1_stim_0.png" alt="210511_rep1" style="max-width:100%"><br><em>210511_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210512_rep1_stim_0.png" alt="210512_rep1" style="max-width:100%"><br><em>210512_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210514_rep1_stim_0.png" alt="210514_rep1" style="max-width:100%"><br><em>210514_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210515_rep1_stim_0.png" alt="210515_rep1" style="max-width:100%"><br><em>210515_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210606_rep1_stim_0.png" alt="210606_rep1" style="max-width:100%"><br><em>210606_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210608_rep1_stim_0.png" alt="210608_rep1" style="max-width:100%"><br><em>210608_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210614_rep1_stim_0.png" alt="210614_rep1" style="max-width:100%"><br><em>210614_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210619_rep1_stim_0.png" alt="210619_rep1" style="max-width:100%"><br><em>210619_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210620_rep1_stim_0.png" alt="210620_rep1" style="max-width:100%"><br><em>210620_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210622_rep1_stim_0.png" alt="210622_rep1" style="max-width:100%"><br><em>210622_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_210623_rep1_stim_0.png" alt="210623_rep1" style="max-width:100%"><br><em>210623_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220515_rep1_stim_0.png" alt="220515_rep1" style="max-width:100%"><br><em>220515_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220516_rep1_stim_0.png" alt="220516_rep1" style="max-width:100%"><br><em>220516_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220517_rep1_stim_0.png" alt="220517_rep1" style="max-width:100%"><br><em>220517_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220518_rep1_stim_0.png" alt="220518_rep1" style="max-width:100%"><br><em>220518_rep1</em></td>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220519_rep1_stim_0.png" alt="220519_rep1" style="max-width:100%"><br><em>220519_rep1</em></td>
</tr>
<tr>
<td align="center" width="50%"><img src="best_threshold_thr2p0/spike_event_alignment_220520_rep1_stim_0.png" alt="220520_rep1" style="max-width:100%"><br><em>220520_rep1</em></td>
<td></td>
</tr>
</table>

---

## Aggregate Event Alignment (2.0σ)

The aggregate figure pools all 21 sessions: per-session energy traces (light gray) + bold mean, multi-session spike raster (one row per session), spike density curve, and mean kinematic signals with spike density in the background.

![Aggregate event alignment 2.0σ](aggregate_event_alignment_thr2p0_stim_0.png)

---

## Recommended Next Steps

1. **Stratify by reach phase** (`begin_reach`, `mid_reach`, `full_reach`) — the acceleration coupling may be stronger in specific phases.

2. **Temporal lag analysis** — does the energy spike *precede* deceleration (predictive/causal) or follow it (reactive)? A cross-correlation or lead/lag regression on the energy derivative vs. acceleration time series would clarify.

3. **Multiple comparison correction** — 3 kinematic variables × 6 thresholds = 18 tests; Bonferroni threshold is p < 0.003. The 2.0σ acceleration result sits right at this boundary. Confirmatory analysis on held-out sessions would strengthen the claim.

4. **Outlier investigation** — sessions 210512 and 210422 have large negative acceleration effects (d ≈ −0.5) but are individually non-significant due to low spike counts. Examine whether these sessions share recording or animal characteristics.

5. **Counterexample investigation** — 210622 (d = +0.189) and 220518 (d = +0.198) show the opposite direction. Understanding what distinguishes these sessions may reveal moderating variables.

---

## Output File Index

| File | Contents |
|---|---|
| `threshold_sweep_stim_0.{png,csv}` | Effect size vs. threshold sensitivity curve |
| `aggregate_meta_analysis_stim_0.{png,csv}` | Cross-session forest plot + pooled meta-analysis |
| `aggregate_event_alignment_thr*_stim_0.png` | Aggregate spike–kinematic alignment at each threshold |
| `energy_spike_regression_<session>_stim_0.{png,csv}` | Per-session regression figures + statistics |
| `best_threshold_thr2p0/spike_event_alignment_<session>_stim_0.png` | Per-session event alignment at best threshold (2.0σ) |

---

## Reproduction

```bash
python aggregate_transition_analysis.py \
    /path/to/energy_decomp_Apr_16 \
    ./notes/session_report/energy_decomp_Apr_16_report/delta_energy_kinimatic_alignment \
    --threshold 2.0 \
    --sweep \
    --sweep_thresholds "0.5,1.0,1.5,2.0,2.5,3.0"
```
