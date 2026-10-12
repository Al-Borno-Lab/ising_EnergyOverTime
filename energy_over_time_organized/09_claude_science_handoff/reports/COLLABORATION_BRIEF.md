# Collaboration brief: Purkinje-cell Ising project (status 2026-10-07)

A self-contained summary for discussing the project with another person or agent. Detailed documents are listed at the end.

## 1. Data
- Calame, Becker & Person (bioRxiv 10.1101/2021.12.17.473247). Mice reach for a target while Neuropixels probes record Purkinje cells (PCs) in Lob 4/5 / Simplex.
- Each session has three blocks of about 20 reaches: baseline (stim_0); closed-loop optogenetic stimulation of pontine mossy fibers (stim_1), triggered when the hand crosses 1 cm outward (50-ms, 100-Hz train); and washout (stim_2).
- 21 sessions from 6 mice: fChR2, fChR4, fChR5, cChR1 (2021) and DJC000, DJC002 (2022; not in the preprint). 3–32 PCs per session.
- Spikes are binned at 10 ms and binarized (spike / no spike). The analysis window is 700 bins around the reach.

## 2. What is established
1. Stationary pairwise Ising (maximum-entropy) fits reproduce the PC population statistics, and the population is more collective than independent neurons predict.
2. In the stationary model, the coupling term ΣJss appears to rise during the reach. That is an artefact: shared firing-rate changes leak into the coupling term.
3. With a time-varying-field model (each neuron's field h_i(t) follows the reach, J fixed), couplings still improve held-out prediction in 21/21 sessions, above a trial-shuffle null. The coupling is present at rest. There is no evidence that J changes during the reach (95% CI on the change: −28% to +93%). The movement-related energy change sits in the fields, i.e. the firing rates.
4. Pipeline verification from Dylan's raw files: extraction, binarization, energy terms and the MCH fit are all correct. **One issue:** 157/1249 reaches flagged `exclude = 1` were never dropped. Refits without them are running now.

## 3. The novel hypothesis (operating-point hypothesis)
**Claim:** PC couplings are fixed, but their *influence* on the population output changes over the reach. PCs fire tonically, with baseline spike probability about 0.60 per 10-ms bin. The reach-related pause brings them toward p ≈ 0.5, where a binary unit is most variable. There, the same fixed couplings shape joint activity patterns most strongly.

**Why it needs a stat-mech model.** A correlation coefficient mixes (i) the couplings, (ii) the operating point (the firing rates), and (iii) their interaction. The Ising model separates them: J fixed, h(t) moving. From (h(t), J) it can compute, at every moment, the multi-information (how much the couplings constrain the population), the susceptibility (sensitivity to a small shared input), and the counterfactual with J = 0 at the same rates.

**Why it is novel.**
- The pairwise version is known in cortex (de la Rocha et al. 2007: correlations rise with rate under fixed shared input). There, the mechanism raises correlations as rates go *up*. In tonic PCs it should raise coupling influence as rates go *down* (the pause).
- It reconciles "synchrony carries reach information" (Nashef et al. 2023) with "couplings do not change": synchrony can change because rates move the population into a more coupling-sensitive regime.
- It is about what the downstream cerebellar nuclei actually receive, i.e. synchronous PC pauses (Person & Raman 2011).
- It is falsifiable (see below).

**Honest caveat.** Part of the effect follows from the mathematics of binary variables. The scientific content lies in whether the effect is large in real PCs, whether it is aligned to the reach, and whether the predictions below hold.

**Predictions / pass–fail.**
- Coupling influence (multi-information, susceptibility, coupling-driven synchronous pauses) peaks during the reach.
- The peak disappears when the fields are frozen at their baseline values.
- Peak size scales with how far the pause moves each population toward 0.5. The one session with baseline < 0.5 should show the opposite.
- Mossy-fiber stimulation should push rates away from 0.5 and reduce coupling influence during the pulse, while J itself stays unchanged.
- **Refuted if:** there is no peak; or the peak survives frozen fields; or the effect is confined to the 2022 mice.

## 4. Steps (from PAPER_ANALYSIS_PLAN.md)
| # | Step | Status |
|---|---|---|
| 0 | Rebuild inputs from raw data; verify the pipeline | **Done** (PIPELINE_VERIFICATION.md) |
| — | Drop the 157 excluded reaches; refit (MCH pipeline + time-varying model) | Time-varying model **done**: conclusions unchanged (coupling above null 21/21 sessions, 62/63 conditions; no reach-phase change). MCH refit **running** on the cluster (08_exclusion_applied) from a copy of the code with two plotting bugs fixed |
| 1 | Per-reach alignment (each reach's own peak velocity / true 1-cm crossing) | Next |
| 14 | Pre-specify metrics, windows and tests before steps 3–6 | Before step 3 |
| 2 | Operating point over time: per-neuron p_i(t), distance to 0.5 | Planned |
| 3 | Time-resolved thermodynamics: multi-information, susceptibility, heat capacity. Uses moment-matched parameters; the plain pseudo-likelihood fit is unreliable in the 2022 sessions | Planned |
| 4 | Controls: frozen fields, epoch-specific J, shuffle null, correlated-pair pruning, latent-factor control, mouse as unit (6 mice) | Planned |
| 5 | Mechanism: peak influence vs operating-point shift | Planned (go/no-go with 3–4) |
| 6 | What the nuclei see: coupling-driven excess of synchronous pauses (J vs J = 0) | Planned (fallback claim) |
| 7–9 | Model adequacy over time; bin-size sweep (Nashef timescale); link to reach velocity | Planned |
| 10 | Stimulation test using true laser times (now available in the raw files) | Planned |
| 11–12 | Complex spikes (reported unreliable; exploratory only); unit positions (channel/depth available) | Optional |

Go/no-go: if the frozen-field control removes the peak and the operating-point relation holds, the central claim stands. Otherwise the paper falls back to the co-silence claim (step 6) or stimulation invariance (step 10).

## 5. Open questions for the lab
- 2022 cohort (DJC000/DJC002): recording site, probe, laser power, spike sorting?
- Was the unit-stability filter applied (the paper kept 126/231 units)?
- Meaning of `ReachS.exclude` / `qm` (assumed to be tracking quality).

## 6. Key files
- PAPER_ANALYSIS_PLAN.md: full plan, figures (7 main, S1–S10), data-requirements table
- PAPER_NOVELTY_CANDIDATES.md: ranked novelty candidates and the literature check
- PIPELINE_VERIFICATION.md: raw-to-fit verification and issues found
- RESULTS_LOG_FOR_PAPER.md: established results and recommended wording
- ISING_FINDINGS_FIGURES.pdf: current figures with captions
- Cluster: energy_over_time_organized/08_exclusion_applied/README.md (git-tracked)
