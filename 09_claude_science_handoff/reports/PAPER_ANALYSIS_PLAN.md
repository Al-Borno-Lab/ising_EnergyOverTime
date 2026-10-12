# Analysis and figure plan: "The reach pause moves Purkinje cells into a coupling-sensitive regime" (2026-10-07)

**Central hypothesis.** Purkinje-cell (PC) couplings J are fixed, but their influence on the population output pattern changes over the reach. The reach-related pause moves tonically active PCs (baseline spike probability about 0.60 per 10-ms bin) toward 0.5, where a binary unit is most variable. There, fixed couplings shape joint patterns most strongly.

**What would support it:** coupling influence (multi-information, susceptibility, coupling-driven synchronous pauses) peaks during the reach; the peak disappears when fields are frozen at baseline; its size scales with how close each population moves to p = 0.5; it is absent in shuffle nulls.
**What would refute it:** no peak in coupling influence; or a peak that does not depend on the operating point (that would point to a J change, contradicting the held-out results); or effects confined to the 2022 cohort.

## A. Steps

### Prerequisites (no new data)
0. **Rebuild the working inputs.** Re-export binarized spikes from the raw .mat files on math-alderaan (previously verified identical to the analysed spikes in 63/63 conditions). Restore tvfield_model.py and run_tvfield.py from artifacts. Rewrite the stimulation model code (stimfit_model.py, run_stimfit.py, validate_stimfit.py; design recorded in memory). Save all code and the spike export as artifacts this time.
1. **Per-reach alignment.** Refit the time-varying-field (TVF) model with the field basis aligned to each reach's own peak velocity, not the condition median. Re-check that A7 (coupling beyond rates) and A8 (no detectable J change) still hold. Keep the per-time-point held-out gains and fitted (h(t), J) per condition as a checkpoint artifact.

### Core analyses (no new data)
2. **Operating point over time.** For each neuron, the model firing probability p_i(t) = σ(2h_i(t) + coupling input) and the independent-model p_i(t). Summaries: distance to 0.5, variance p(1−p), and the fraction of neurons within ±0.1 of 0.5, all over time. Also the per-neuron field-energy sign check (open item).
3. **Time-resolved thermodynamics.** For each condition and 10-ms bin t, from (h(t), J):
   - entropy S_Ising(t) and S_indep(t), and multi-information I(t) = S_indep − S_Ising (bits per neuron);
   - susceptibility χ(t) = d⟨Σs⟩/dε to a small uniform field ε;
   - heat capacity C(t) = Var(E) under the model at that t;
   - model-predicted pairwise correlations, with vs without J.
   Exact enumeration for N ≤ 20 (39 conditions); Gibbs/MCMC for larger N, validated against exact results at N ≤ 20.
4. **Coupling counterfactuals and controls.**
   - (a) **Frozen-field control:** recompute I(t), χ(t) with h held at its baseline value. The hypothesis predicts no reach modulation.
   - (b) **Frozen-J control:** J fixed by construction. Compare with epoch-specific J (run_epochJ fits) to show the time course is not a J change.
   - (c) **Trial-shuffle null:** refit on shuffled data and recompute everything. Expect I(t) ≈ 0.
   - (d) **High-correlation pruning** (drop one unit of every pair with r > 0.5).
   - (e) **Latent-factor control:** add 1–2 shared per-reach factors to the fields.
   - (f) **Mouse/cohort:** mixed model with mouse as a random effect (6 mice), plus the 2021 and 2022 cohorts reported separately.
5. **Mechanism test across neurons and sessions.** Regress the peak reach-related change in I(t) or χ(t) on the change in mean p(1−p) (equivalently, how far the pause moves the population toward 0.5).
   - Within session: neurons that cross 0.5 vs neurons that stay far from it.
   - Across sessions: deeper pauses toward 0.5 should give larger peaks.
   - The one session with baseline < 0.5 is a natural opposite-direction case.
6. **What the cerebellar nuclei receive.** Probability of K or more PCs silent together, P(K_silent ≥ k | t), from the model with J vs J = 0 at the same h(t). Report the ratio over time; this is the coupling-driven synchronous-pause excess.
   - Model-free check on held-out data: observed synchronous pauses compared with the time-varying independent prediction.
   - Co-silence vs co-firing split of the coupling influence.
7. **Model adequacy over time.** Surprise = E_observed(t) − ⟨E⟩_model(t) on held-out reaches. Systematic deviation during the reach would mean structure the model misses (higher-order terms or a J change).
8. **Timescale.** Bin-size sweep (5, 10, 20 ms; 2 ms if sparsity allows) of the held-out coupling gain and of the I(t) peak. This addresses Nashef et al. (millisecond synchrony).
9. **Behavioral link.** Per reach: coupling influence in the reach window vs peak velocity and endpoint. The dataset paper reports that the population pause scales with velocity, so faster reaches should show larger coupling influence.

### Needs new data from the user
10. **Stimulation test** (stimulation timing or unscaled kinematics). The joint per-session model (shared J, block-specific fields, κ in the pulse window), plus I(t) and χ(t) in the pulse window. Prediction: extra mossy-fiber drive pushes rates up, away from 0.5, so coupling influence falls during the pulse while κ ≈ 0.
11. **Complex spikes** (if CS times are available). Test whether CS-triggered pauses account for the co-silence coupling.
12. **Unit positions** (if available). J and coupling influence as a function of distance (microzones).

### Writing and literature
13. **Literature positioning:** de la Rocha et al. 2007 (correlations rise with rate under fixed shared input); Shea-Brown et al. 2008; Granot-Atedgi et al. 2013 and Shimazaki et al. 2012 (time-dependent maxent); criticality pitfalls (Schwab, Nemenman & Mehta 2014; Aitchison, Corradi & Latham 2016); Person & Raman 2011; Ramirez & Stell 2016; Nashef et al. 2023; Calame et al. (dataset).
14. **Pre-specify** the hypothesis, metrics, windows and tests before running steps 3–6 on real data, and freeze them in this document. This protects against the window-choice problems found earlier.

## B. Figures

### Main figures
| Fig | Content | Steps |
|---|---|---|
| 1 | **Concept and data.** (a) Reach task and per-reach kinematics; (b) example PC raster with the reach pause; (c) TVF Ising schematic: h(t) follows rates, J fixed; (d) toy two-neuron example: for fixed J, multi-information and correlation as a function of firing probability, peaking near p = 0.5 | 0, 1 |
| 2 | **Fixed coupling beyond rates.** Held-out coupling gain vs trial-shuffle null over time (per neuron, per session, coloured by cohort); by phase; exact vs pseudo-likelihood; epoch-specific J gives no gain | 1, 4b |
| 3 | **The operating point moves.** (a) per-neuron p_i(t) heatmap aligned to peak velocity; (b) population distribution of p over time crossing 0.5; (c) fraction of neurons near 0.5 and mean p(1−p) over time | 2 |
| 4 | **Coupling influence peaks during the reach.** I(t), χ(t), C(t) over time with the frozen-field control and the shuffle null; per-session peaks; cohort breakdown | 3, 4 |
| 5 | **Mechanism.** Peak influence vs operating-point shift across sessions and neurons (crossing 0.5 vs not); prediction checks including the baseline-below-0.5 session | 5 |
| 6 | **What the nuclei see.** Coupling-driven excess of synchronous pauses P(K_silent ≥ k) over time (J vs J = 0 at matched rates); model-free held-out check; co-silence vs co-firing | 6 |
| 7 | **Perturbation test (if data allow).** Pulse-window rates, coupling influence and κ across baseline / stim / washout and early vs late stimulated reaches | 10 |

### Supplementary figures
| Fig | Content | Steps |
|---|---|---|
| S1 | Synthetic validation: fixed J with moving h recovers the I(t) peak; a true J change is distinguishable from an operating-point shift; MCMC vs exact | 3 |
| S2 | Per-reach vs condition-level alignment: A7/A8 unchanged | 1 |
| S3 | Stationary-model pitfall: rate changes leak into ΣJss (current Figs. 6–7) | — |
| S4 | Structure beyond independence: P(K) falsification and surrogate refits (current Figs. 2–3) | — |
| S5 | Robustness: pruning, latent-factor control, mouse mixed model, cohort split | 4d–f |
| S6 | Model surprise over time | 7 |
| S7 | Bin-size sweep | 8 |
| S8 | Reach-by-reach link to kinematics | 9 |
| S9 | Reach landmarks QC (current Fig. 1) and phase definitions | — |
| S10 | Complex-spike and spatial analyses (if data allow) | 11, 12 |

## Data requirements and answers so far (2026-10-07)
| Need | Status | Consequence |
|---|---|---|
| Stimulation timing | Paper: closed-loop trigger at the 1-cm outward crossing, 9.5 ms latency, 50-ms train (100 Hz, 2-ms pulses). Reaches are 10-s segments centred on that crossing, so the pulse starts about 1 bin after each segment's midpoint | Usable now as segment midpoint + 1 bin; confirm against ReachS.filt_kin (unscaled cm) when the source files arrive |
| Opsin controls | All mice received AAV-ChR2 in the pontine nuclei; the controls were red-light sessions | All sessions can test the stimulation prediction |
| Complex spikes | Present (CS_Bin1, 1-ms bins) but described to the user as unreliable; the paper could identify them only in some cells | Step 11 is exploratory at most; do not build claims on it |
| PC identity | Only cellID == "PC" units are exported. PCs were identified by CS pause where possible, otherwise by rate > 40 sp/s, CV2 > 0.20, MAD < 0.008 | Some units are putative PCs; state this in Methods |
| Unit stability filter | Unknown whether the paper's waveform/drift filter was applied | Check in the source files; until then, include a per-unit stability check before any cross-block claim (step 10) |
| Unit positions | Probably derivable from cellData / Kilosort output | Step 12 waits for the source files |
| 2022 cohort (DJC000, DJC002) | Not in the preprint; protocol unknown; the user will ask the lab | Report the 2022 sessions as a separate group until the protocol is confirmed |
| Spike counts | Bin10 holds counts per 10-ms bin (max 11); the pipeline binarizes them as > 0 | Note it in Methods; check bins with ≥ 2 spikes as a sensitivity analysis |
| Source files (cellData, ReachS) | User is downloading | Unlocks unscaled kinematics, exclude flags, unit positions and stability |

## C. Order of work
1. Steps 0–1 (rebuild, re-align): about half a day of compute.
2. Pre-specify (step 14), then S1 synthetic validation, then steps 2–3 on real data.
3. Steps 4–7 (controls, mechanism, the nuclei's view, adequacy).
4. Steps 8–9.
5. Steps 10–12 when data arrive.
6. Assemble figures, then write.

Steps 3–5 are the go/no-go point. If the frozen-field control removes the peak and the operating-point regression holds, the paper's central claim stands. If not, fall back to the co-silence claim (step 6) or the stimulation-invariance claim (step 10).
