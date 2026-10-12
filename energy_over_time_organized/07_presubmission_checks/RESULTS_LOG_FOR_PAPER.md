# Results log for manuscript (updated 2026-09-29; pre-submission checks closed)

This log sorts the results of the Purkinje-cell pairwise maximum-entropy (Ising) analysis into established (A), robust (B), suggestive (C) and withdrawn (D). Every number comes from a saved artifact. Details of the pre-submission checks are in OPEN_ITEMS_RESOLUTION.md. Cluster tree: `math-alderaan.ucdenver.pvt:/home/enserrog/ising_EnergyOverTime/energy_over_time_organized/` (new checks are in `07_presubmission_checks/`).

**Dataset.** 21 sessions × 3 stimuli = 63 conditions, 3-32 Purkinje cells each, 1,249 reaches. Bins are **10 ms** (confirmed from the raw .mat time column); 700 bins per reach. The unit of replication is the session (n = 21). **Model convention:** E = −(ΣJ_ij s_i s_j + Σh_i s_i), with s ∈ {−1, +1}.

## A. Established

| # | Result | Key numbers | Source |
|---|---|---|---|
| A1 | The pipeline is correct | Bit-exact reproduction for all 9 sessions with raw data (27 conditions). A fresh refit reproduces the stored J (r ≥ 0.998) | VERIFICATION_REPORT.md, solver_validation.csv |
| A2 | Populations show collective structure beyond stationary independent firing (model-free) | P(K) differs from the independent-Bernoulli null in 27/27 conditions (z = 4.9 to 116) | falsification_findings.md |
| A3 | Fitted couplings are not a fitting artefact | Refits on 100 rate-preserving shuffles per condition: real mean \|J\| exceeds every surrogate in 63/63 conditions (median 4.6x, min z = 3.6) | surrogate_refit_results.csv |
| A4 | Pairwise model beats the constant-rate independent model on held-out data | 5-fold cross-validation over reaches: 21/21 sessions, 61/63 conditions, median +0.21 nats per bin | timeresolved_llr_summary.csv |
| A5 | Pairwise structure beyond time-varying rates exists at rest | Ising beats a time-varying-rate independent model in the pre-reach baseline in 16/21 sessions (p_FDR = 0.004) | timeresolved_llr_summary.csv |
| A6 | J can change from correlation structure alone | Synthetic two-population simulation | 06_synthetic_validation_simulation |

A2 and A3 use shuffles that also remove reach-locked rate modulation. They therefore establish structure beyond *stationary* independence; the separation of coupling from shared rate modulation comes from A5 and B10.

## B. Robust within this dataset (hold across windows, BH-FDR, and with 210620/220515 excluded or deduplicated)

| # | Result | Key numbers | Source |
|---|---|---|---|
| B1 | The outward reach is brief | Acceleration phase median 90 ms, deceleration phase median 120 ms | phase_landmarks.csv |
| B2 | The movement-related energy drop is carried by the coupling term | Coupling ΔE below baseline in acceleration (p = 1e-5), deceleration (p = 1e-4) and post-movement (p = 7e-4). Field term p ≥ 0.84. Unchanged after deduplication | energy_decomposition_by_phase.csv, dedup_effect.csv |
| B3 | J is a broad elevation spanning the reach, not a timed peak | Session-mean J (z) is above 1 from −340 ms, crosses half-maximum at −200 ms, peaks (z = 4.7) at +430 ms, and returns below half-maximum at +1,050 ms relative to peak velocity | J_mean_timecourse.csv |
| B4 | No signal follows acceleration within a phase or across reaches | All p_FDR > 0.05 | phase_correlations_summary.csv |
| B5 | All four signals are locked to the movement, with a sign that varies across sessions | Alignment z 1.2 to 1.6 vs a circular-shift null, 18-20/21 sessions | phase_correlations.csv |
| B6 | Firing rate is more movement-modulated than any Ising quantity | FR wins 15-18/21 sessions (p_FDR = 0.003-0.044) | arbitration_vs_firing_rate.csv |
| B7 | Single-reach J maxima cluster at peak velocity | 12.4% within ±50 ms vs 6.4% in pre-reach control windows (p < 1e-14), concentrated in 220516, 210425, 210608 and 210512 | J_peak_times_single_reach.csv |
| B8 | Energy and field-term variance dip after the reach | Troughs at +310 to +380 ms, p_FDR ≤ 0.02 at every window size | sensitivity_summary.csv |
| B9 | J modulation without firing-rate modulation is rare | ρ = 0.74 across 63 conditions; one clear example (210511 stim 1) | FLAT_FR_J_DISSOCIATION.md |
| B10 | The movement-time Ising advantage comes from reach-locked rate modulation | Against a time-varying-rate independent model, Ising is **worse** during acceleration (2/21 sessions, p_FDR = 0.004) and not different during deceleration or post. Against a constant-rate model its advantage rises in 20/21 sessions | timeresolved_llr_summary.csv, coupling_source_checks.png |
| B11 | The J-term rise is partly rate-predicted, with a mix that varies by session | Only a median 36% comes from across-reach covariance. The covariance part carries the rise in sessions where FR falls (ρ = −0.66), the rate part where FR rises (ρ = +0.64). Neither part alone is consistent (p_FDR > 0.16) | j_rate_vs_cov_summary.csv |
| B12 | Timing of J, h and firing-rate maxima matches chance | Real vs control-window lag not different in any of 6 window/exclusion variants | peak_timing_vs_control_all_variants.csv |

**How B2, B10 and B11 fit together.** The coupling term drives the energy drop during the reach. A large share of that drop follows from the population's shared rate trajectory, evaluated through the fitted couplings, rather than from coordination beyond rates. Write it as "the network moves into low-energy, coupling-favoured states during the reach", not as "neurons become more coordinated during the reach".

## C. Suggestive only (do not state as findings)
- **Late covariance rise:** 500-1500 ms after peak velocity, the across-reach covariance part of J rises (13/21, p_FDR = 0.053), and Ising beats the time-varying-rate independent model on average (12/21, p = 0.07). This is a candidate post-movement coordination epoch carried by a subset of sessions.
- **Acceleration-phase J maxima:** 10/63 conditions, against about 4 expected from phase length. Borderline after FDR.
- **Early energy maxima:** energy maxima come earlier than control (−350 ms, p_FDR = 0.017), but only in 2 of 6 window variants.
- **aggregate_transition_analysis:** energy-spike bins have lower acceleration in stim 0 only (d = −0.103, p_FDR = 0.021); stims 1-2 are not significant.
- **Partial flat-FR example:** 210619 stim 2.

## D. Withdrawn or not supported

| Claim | Status | Reason |
|---|---|---|
| "J peaks ~300 ms after peak acceleration" | Withdrawn | Asymmetric search window. Against control windows: +300 vs +240 ms, p = 0.24 |
| "J lags acceleration in ~70% of sessions" (arbitration_j_many) | Withdrawn | Regenerated: 40/63. Control windows give 75%; real vs control p = 0.79 |
| "J is the best indicator of peak acceleration" (test_collective_j_timing) | Withdrawn | Unbiased analysis: no indicator is closer than any other (all p ≥ 0.41). J-gated: FR closer (p = 0.042) |
| Pooled Cohen's d = −0.114, p = 0.003 | Replaced | See C: stim 0 only, d = −0.103 |
| "Neurons coordinate more during the reach, beyond firing rate" | Not supported | B10, B11 |
| "Most J peaks fall after the movement" / "in the deceleration phase" | Not supported | Window-length artefact; deceleration phase is only 120 ms |
| Collective state locked to arm acceleration (PMC10370034 framing) | Not reproduced | B4, B12 |
| Criticality proxy: Var(J) peaks +210 ms | Did not replicate | +90 / +110 / +20 ms at W = 15/25/40, none significant after FDR |
| Decision tree predicting J peaks | Abandoned | Leave-one-session-out accuracy ≤ majority baseline |

## E. Remaining open items
1. **Time-varying-field Ising model:** fit h_i(t) with J fixed and compare against the time-varying independent model. This is the direct test of whether pairwise coupling adds anything beyond shared rate modulation *during* the reach (B10 leaves it open).
2. **Code fixes (not yet applied to project code):**
   - `aggregate_transition_analysis.py` session labels break under the organized layout (all sessions become `full_reach`);
   - `pk_j_analysis.py` line 449 needs Python ≥ 3.12, so it fails in the project container;
   - the timing scripts need `shared/` on PYTHONPATH.
3. **Late post-reach epoch:** confirm the late covariance/Ising epoch (C1) with a pre-registered window.

## F. Methods conventions
- **Kinematics:** x position, Savitzky-Golay filter (11 bins, order 3). Landmarks: onset (10% of peak velocity), peak acceleration, acceleration zero-crossing (= peak velocity), peak deceleration, deceleration end. Baseline −1,000 to −400 ms (decomposition) or −2,000 to −400 ms (z-scoring).
- **Statistics:** session-level Wilcoxon tests with BH-FDR within each family. Extremum timing is always compared against the same search in control windows shifted −1.5, −2 and −2.5 s.
- **Fitting:** coniii MCH (max_iter 75, learning rate 1/(i+1), chain max(100N, 2000)), run in `inverse-ising-arm-2.sif`.
- **Model comparison:** 5-fold cross-validation over reaches. log Z computed exactly for N ≤ 22, by thermodynamic integration above that (error ≤ 0.004 nats).
- **Surrogates:** independent per-neuron permutation, 100 per condition.

## G. Figures

| Figure | Supports | Status |
|---|---|---|
| J_timing_rise_decay | B3, B7 | Use |
| energy_decomposition_by_phase | B2 | Use |
| coupling_source_checks | B10, B11, C1 | Use; key interpretive figure |
| surrogate_refit_J | A3 | Use (supplement) |
| peak_timing_vs_control | B12 | Use; supersedes phase_timing_summary panel b and the sensitivity_robustness peak rows |
| flat_FR_vs_J_modulation + example_flatFR_210511_stim1 | B9 | Use |
| phase_landmarks_qc | B1 | Methods / supplement |
| phase_timing_summary | B4, B6 (panels a, c, d) | Drop panel b |
| per_session_phase_timing_figures | — | Supplement |

### Key figures

![Coupling source](/Users/gunnarenserro/.claude-science/orgs/f552423d-f1dd-4c5d-879b-edcfa3506e26/artifacts/proj_aa374d75609c/f79ff9ab-6cc1-4959-b3b8-21436f0032a4/v6526a49c_coupling_source_checks.png)

![Energy decomposition](/Users/gunnarenserro/.claude-science/orgs/f552423d-f1dd-4c5d-879b-edcfa3506e26/artifacts/proj_aa374d75609c/99fe21b3-98be-43e4-ab8c-c9144dbe2f2d/vd6234ab1_energy_decomposition_by_phase.png)

![J timing](/Users/gunnarenserro/.claude-science/orgs/f552423d-f1dd-4c5d-879b-edcfa3506e26/artifacts/proj_aa374d75609c/88c084a6-efe1-4709-aa5b-fa1b91c8fe78/v2f8659f8_J_timing_rise_decay.png)

![Peak timing vs control](/Users/gunnarenserro/.claude-science/orgs/f552423d-f1dd-4c5d-879b-edcfa3506e26/artifacts/proj_aa374d75609c/933e4afc-1d96-4261-95e2-0d910cc464a2/v8b898ef5_peak_timing_vs_control.png)
