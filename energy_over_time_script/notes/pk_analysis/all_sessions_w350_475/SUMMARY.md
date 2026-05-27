# P(K) × J-Peak Analysis — Summary

**Dataset:** `energy_decomp_may12_stimDecon`
**Window:** bins 350–475 (search region around reach onset)
**Sessions:** 21 | **Stim/session pairs:** 42 (stim 0 & 1 only — stim 2 absent in this dataset)
**Peak threshold:** 1.75

---

## Hypothesis

A J coupling peak around reach onset co-occurs when **both** of the following hold:

1. **Ising fits the data better than the independent model** (`r_ising > r_independent`)
2. **Ising and independent model distributions diverge** (`ising_indep_dist > 0.1`)

When both conditions are met, pairwise neural correlations (J) are genuinely necessary to explain the population activity — neurons are in a collectively coordinated state, not just independently active.

---

## Statistical Result

|  | Hypothesis MET | Hypothesis NOT MET | Total |
|---|---|---|---|
| **J peak present** | 21 | 7 | 28 |
| **No J peak** | 5 | 9 | 14 |

**Chi-squared test:** χ² = 4.556, df = 1, **p = 0.033**

The association is statistically significant. When a J peak occurs, the hypothesis is met **75%** of the time. When no peak occurs, it is met only **36%** of the time.

---

## Key Metric Differences

| Metric | J peak sessions | No-peak sessions |
|---|---|---|
| `r_ising` | 0.959 ± 0.067 | 0.967 ± 0.083 |
| `r_independent` | **0.890 ± 0.168** | **0.972 ± 0.063** |
| `r_ising_vs_indep` | 0.887 ± 0.168 | 0.978 ± 0.042 |
| `ising_indep_dist` | **0.370 ± 0.284** | **0.181 ± 0.139** |

The Ising model fits about equally well regardless of whether a peak occurs (`r_ising` ~0.96 in both groups). The critical difference is that the **independent model fits worse** in peak sessions (0.890 vs 0.972), and **Ising and independent diverge more** (0.370 vs 0.181). This means J peaks are not about Ising getting better — they are about the independent model *failing*, i.e., individual firing rates alone cannot explain the population activity during the peak.

---

## Exceptions and Anomalies

### False positive detections (hypothesis not met despite detected peak)

These sessions had a detected J peak but near-zero `ising_indep_dist`, meaning Ising and independent models are nearly identical. Visual inspection confirmed the "peaks" are embedded in noise rather than representing real collective events.

| Session | Stim | Peak ratio | `ising_indep_dist` | Notes |
|---|---|---|---|---|
| 210423 | 1 | 2.783 | 0.026 | Ising ≈ Independent — no collective signature |
| 210622 | 0 | 1.837 | 0.046 | Same; borderline detection |
| 210622 | 1 | 2.953 | 0.023 | Same |
| 210511 | 1 | 1.850 | 0.067 | Near-threshold dist |

**Implication:** `ising_indep_dist` can be used as a second filter beyond the time-series peak detector. A meaningful J peak should satisfy both the peak detection criterion **and** `ising_indep_dist > 0.1`.

### Genuine exception — 220515

| Session | Stim | Peak ratio | `r_ising` | `r_independent` | `ising_indep_dist` |
|---|---|---|---|---|---|
| 220515 | 0 | 5.310 | 0.941 | **0.950** | 0.459 |
| 220515 | 1 | 6.595 | 0.946 | **0.961** | 0.440 |

220515 has strong, visually clear J peaks and large Ising–independent divergence, yet the independent model outperforms Ising on the P(K) fit. This is the true anomaly. One possible explanation: this session has unusually strong single-neuron drives (h fields) that dominate the population activity distribution, so the independent model captures P(K) well even though pairwise interactions are active in the time domain. The J peak reflects coordinated *dynamics* that do not leave a clear footprint in the time-averaged P(K) distribution.

---

## Interpretation

The results support the hypothesis that J peaks around reach onset reflect genuine collective neural activity. The mechanism appears to be:

- In **collective sessions** (peak present): individual firing rates alone cannot predict how many neurons are simultaneously active — the pairwise correlations captured by J are necessary, and the Ising model diverges substantially from the independent baseline.
- In **non-collective sessions** (no peak): the independent model explains the population distribution nearly as well as Ising, suggesting neurons are largely acting independently around the reach.

The finding is not that Ising gets *better* at reach time — it's that the **independent model gets worse**, which means pairwise interactions become load-bearing.

---

## Caveats

1. **Small dataset:** 42 session/stim pairs across 21 sessions. The chi-squared test is significant but the effect size should be replicated with more data.
2. **Stim 2 absent:** This dataset only contains stim 0 and 1 per session. Results may differ for other stimulus conditions.
3. **Peak label quality:** Several sessions have borderline detections (`ising_indep_dist` near zero) that appear to be noise rather than real collective events. A combined criterion (time-series detection AND `ising_indep_dist > 0.1`) would produce cleaner labels.
4. **220515 remains unexplained:** This session consistently violates the hypothesis despite strong J peaks and should be inspected further (e.g., h-field magnitudes, neuron count, recording quality).

---

## Output Files

| File | Description |
|---|---|
| `pk_hypothesis_scatter.png` | Scatter of `ising_indep_dist` vs `r_ising − r_independent`, coloured by J peak status. Points in the top-right confirm the hypothesis. |
| `pk_fit_bars.png` | Bar chart comparing mean `r_ising` and `r_independent` for peak vs no-peak sessions. |
| `leads/stim{N}_{session}.png` | 6-panel per-session plots where J leads kinematics. Panel 6 shows P(K) curves. |
| `lags/stim{N}_{session}.png` | Same for sessions where J lags kinematics. |
| `no_peak/stim{N}_{session}.png` | Sessions with no detected J peak. |
| `pk_summary.txt` | Full per-session table with all metrics and chi-squared test. |

---

## Suggested Next Steps

1. **Apply dual-criterion labelling:** require both peak detection ratio > threshold AND `ising_indep_dist > 0.1` before calling a session a true J peak. Re-run the decision tree with these cleaner labels.
2. **Investigate 220515:** check h-field magnitudes (`h_parameters.csv`) to test whether strong external drive explains the independent model advantage.
3. **Replicate with more sessions/stims** when additional data is available.
4. **Compare across datasets:** run `pk_j_analysis.py` on `energy_decomp_Apr_16` (3-stim dataset) to check if the same pattern holds across stimulus conditions.
