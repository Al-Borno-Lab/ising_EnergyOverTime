# Evidence package: fitting the time-varying-field Ising model to Purkinje cells

Built 2026-10-27. Figures come from the repository and experiment branches, plus exploratory analyses regenerated on 2026-10-27 by `evidence/src/` (branch `analysis/evidence-package`). Pairwise maximum-entropy (Ising) model with time-varying fields h_i(t) (B-splines) and fixed couplings J; 21 sessions, 6 mice, 10-ms bins, 63 session x stimulus conditions. The authoritative claim list is `ising_wt_protocol/RESULTS_LOG_FOR_PAPER.md` (origin/main, updated 2026-10-13); statuses below follow it.

**Status key.** established = log section A; robust = log section B; preregistered-null = preregistered test not supported; inconclusive = preregistered outcome (c); exploratory = not preregistered, not in the log; withdrawn = log section D or retracted.

**PDF key.** "vector" = the PDF was copied from the source (or split page-by-page from a source PDF). "rasterized" = no source PDF existed, so the PNG was wrapped in a PDF at full resolution with its dpi preserved (img2pdf).

**Combined file.** `ALL_FIGURES.pdf` h## Regenerated material (2026-10-27)

The first build of this package (earlier the same day) found the local scratch folders that held the exploratory figures empty, so folders S2-S6 and several panels were missing. They have now been regenerated from saved cluster outputs and new cluster runs (SLURM array 4548957, one job per session, for absolute energy, signal overlays, field-dip shuffles and burster/pauser timing):
- `evidence/src/make_figures.py`: all regenerated figures and tables; `regenerated_stats.json` holds every number quoted from them.
- `evidence/src/m0_vs_m1.py`: S6.
- `evidence/src/regen_session.py`: the per-session cluster job.
- `evidence/src/package/`: this INDEX, `ALL_FIGURES.pdf` and the zip.

Figures marked "regenerated" were redrawn by new code, not copied from the lost files. Their numbers were checked against the values reported before the loss; all match except where noted under **Corrections**. Large intermediate files (`.npz`) are kept outside git, in `code/evidence_data/`.


## 1_model_and_validation

**Claim.** The time-varying-field Ising model (B-spline fields h_i(t), fixed J) recovers known rates and couplings on synthetic data, adds held-out gain only when J is real, and the stored pipeline reproduces the data.

**Status.** established (methods)

**Tables copied here:** `tvfield_validation.csv`, `tvfield_validation_vs_MCH.csv`, `solver_validation.csv`


### 01_tvfield_walkthrough.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/tvfield_walkthrough.png
- **Axes:** Top: P(spike) per 10-ms bin vs ms from peak velocity (true p(t) black, fitted sigmoid(h(t)) dashed). Middle: spike raster of 8 of 60 simulated reaches. Bottom: value of every 3rd cubic B-spline basis function vs time.
- **Pattern:** The fitted field follows the planted reach-locked dip in firing probability; the basis is dense (50-ms knots) near the reach and sparse elsewhere.
- **Why it supports the claim:** Shows how h(t) is built and that it can recover a time-varying rate from 60 reaches.
- **Key numbers:** The figure title says 66 cubic B-splines; the code (tvfield_model.bspline_basis) builds 67. The title is off by one; 67 is correct.
- **PDF:** rasterized from PNG (1350x1350 px at 150 dpi)


### 02_tvfield_validation.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/timevarying_field/tvfield_validation.png; PDF: ising_wt_protocol/07_presubmission_checks/timevarying_field/tvfield_validation.pdf
- **Axes:** (a) Held-out gain over the no-coupling model M0 (nats/bin) for three synthetic cases (J = 0; true J; J = 0 with a shared per-reach rate gain); bars = couplings (M1 - M0), 200-ms-lagged states, trial-shuffle null. (b) Real data: J from the pseudo-likelihood fit vs the stored MCH fit for 3 conditions.
- **Pattern:** Gain appears only when J is real; the lagged control and shuffle null are at or below 0. PL and MCH couplings lie on the identity line except one neuron firing in 99% of bins (210511).
- **Why it supports the claim:** The coupling gain is not produced by shared rate fluctuations, and the PL estimator agrees with the established MCH solver.
- **Key numbers:** dPL(M1-M0): true J +0.107 nats/bin; J = 0 -0.002; J = 0 + per-reach gain -0.001. J recovery r = 0.978 (tvfield_validation.csv). PL vs MCH r_J 0.995 for 210421 (tvfield_validation_vs_MCH.csv).
- **PDF:** vector (copied from source)


### 03_pipeline_verification.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/pipeline_verification.png
- **Axes:** (a) Stored-fit error in pair correlation (rms) vs data covariance (rms over pairs), log-log, coloured by cohort. (b) Covariance reproduced (%) vs N. (c) Error of a static PL refit vs stored-fit error.
- **Pattern:** Stored fits reproduce pair correlations with errors far below the data covariance; >= 92% of covariance reproduced in every condition (median 99.98%); a static PL refit fails in the strongly coupled 2022 sessions.
- **Why it supports the claim:** Supports A1 (the stored stationary fits are correct) and documents where PL alone is unreliable.
- **Key numbers:** Covariance reproduced >= 92% in every condition, median 99.98% (from figure).
- **PDF:** rasterized from PNG (2105x866 px at 300 dpi)


### 04_phase_landmarks_qc.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/phase_landmarks_qc.png
- **Axes:** Mean x-velocity (top) and x-acceleration (bottom) for 63 session x stim conditions vs samples (10 ms) from peak velocity.
- **Pattern:** Acceleration ~9 samples, deceleration ~12 samples: the outward reach is brief.
- **Why it supports the claim:** Defines the kinematic landmarks used everywhere (B1).
- **Key numbers:** Acceleration median 90 ms, deceleration median 120 ms (RESULTS_LOG B1).
- **PDF:** rasterized from PNG (1600x1200 px at 200 dpi)


### 05_synthetic_validation_key_measure.png / .pdf
- **Source:** evidence/figures/1_model_and_validation/synthetic_validation_key_measure (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/1_model_and_validation/synthetic_validation_key_measure.pdf
- **Axes:** SYNTHETIC (E09 H0). Estimated vs true change in multi-information, reach vs rest (bits per cell), fixed-coupling model; colour/marker = N cells x reaches; hollow = scenario C (couplings truly changed).
- **Pattern:** Rate-driven scenarios lie on the identity line; scenario C is reported as ~0, because the fixed-coupling measure cannot see a coupling change by design.
- **Why it supports the claim:** The key E10 measure is accurate when rates drive the change, and its blind spot is known.
- **Key numbers:** Sign correct / within 50% of truth: A 20/20 and 20/20, B 20/20 and 20/20 (= the HANDOFF's 40/40); E 18/20 and 17/20; D has a true change of exactly 0 (sign undefined); C 0/20 by design. Redraw of panel a of 5/03.
- **PDF:** vector (copied from source)


## 2_coupling_beyond_rates_at_rest

**Claim.** Purkinje-cell populations show pairwise coupling beyond their shared, reach-locked firing-rate modulation, and this coupling is already present at rest.

**Status.** established

**Tables copied here:** `surrogate_refit_results.csv`, `tvfield_cv_scores.csv`, `tvfield_phase_summary.csv`, `trial_shuffle_null.csv`, `pruned_gain_summary.csv`, `timeresolved_llr_summary.csv`


### 01_PK_beyond_independence.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/falsification_test.png
- **Axes:** Left: P(K), probability that K neurons are co-active in a bin, observed vs independent-neuron prediction (210614 stim 0, N = 25). Right: z-score of the real deviation vs the shuffled-null distribution for each of 27 session x stim conditions; dashed line z = 3.
- **Pattern:** Observed P(K) is wider than the independent prediction (more synchronous and more co-silent bins); every condition exceeds z = 3.
- **Why it supports the claim:** Model-free evidence of collective structure beyond stationary independent firing (A2).
- **Key numbers:** 27/27 conditions, z = 4.9 to 116 (A2). The brief's '21/26 K values outside the chance band in the example' is not shown in this figure; that number is reproduced in 04_P_K_difference_210614 below.
- **PDF:** rasterized from PNG (2833x1247 px at 300 dpi)


### 02_surrogate_refit_J.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/assets/figures/surrogate_refit_J.png; PDF: ising_wt_protocol/07_presubmission_checks/assets/figures/surrogate_refit_J.pdf
- **Axes:** (a) Mean |J_ij| of the refitted model per condition (sorted), real data (dots) vs range over 100 rate-preserving shuffled surrogates (grey bars), log y. (b) Net coupling sum J_ij as z vs surrogates, vs population size N.
- **Pattern:** Real |J| sits above every surrogate in all 63 conditions; net coupling collapses on surrogates in 62/63 (exception 210622 stim 2, N = 3).
- **Why it supports the claim:** Fitted couplings are not a fitting artefact (A3).
- **Key numbers:** 63/63 conditions; median 4.6x, min z = 3.6 (A3).
- **PDF:** vector (copied from source)


### 03_tvfield_vs_independent.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/timevarying_field/tvfield_vs_independent.png; PDF: ising_wt_protocol/07_presubmission_checks/timevarying_field/tvfield_vs_independent.pdf
- **Axes:** (a) Held-out gain over the time-varying-rate independent model (nats/bin; median, IQR over 21 sessions) vs ms from peak velocity: couplings with simultaneous states, 200-ms-lagged control, trial-shuffle null. (b) Gain minus shuffle null by phase (baseline, acc., dec., post, late). (c) Energy change per neuron vs baseline: coupling term (stationary and time-varying models) and field term (time-varying). (d) Gain from exact likelihood vs pseudo-likelihood (39 conditions, N <= 20).
- **Pattern:** Couplings add prediction at every time point including the pre-reach baseline, with no rise during the reach; the lagged control recovers a small fraction; with time-varying fields the energy dip is in the field term; PL and exact gains agree.
- **Why it supports the claim:** Main figure for the coupling question (A7, A8, B2 note): coupling exists beyond shared rates, is present at rest and does not detectably change during the reach.
- **Key numbers:** Gain over shuffle null 21/21 sessions, median 0.259 nats/bin, p_FDR 8e-6 (tvfield_phase_summary.csv); 63/63 conditions above the null 95th percentile (A7). Baseline 20/21. Lagged control 0.051 vs 0.259 (about 1/5; RESULTS_LOG: 0.05 vs 0.18, about 1/4). Exact LL agrees 38/39. rho = 0.999 PL vs exact: not found in a saved table (unverified; do not cite).
- **PDF:** vector (copied from source)


### 04_P_K_difference_210614.png / .pdf
- **Source:** evidence/figures/2_coupling_beyond_rates_at_rest/P_K_difference_210614 (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/2_coupling_beyond_rates_at_rest/P_K_difference_210614.pdf
- **Axes:** Session 210614 baseline, N = 25. (a) P(K) observed vs independent cells with the same rates (Poisson-binomial), linear. (b) Same, log scale. (c) Observed minus independent, with the 95% range of 200 within-cell shuffles.
- **Pattern:** The data have more all-quiet and more many-together bins than independent cells; the middle K values are depleted.
- **Why it supports the claim:** Data-only (no model fitted) picture of collective structure; complements 01_PK_beyond_independence.
- **Key numbers:** 21/26 K values outside the shuffle band (verified). The shuffle also removes reach-locked rate modulation, so this panel alone does not separate coupling from shared rate changes; 03_tvfield_vs_independent does.
- **PDF:** vector (copied from source)


## 3_rates_vs_coupling_over_reach

**Claim.** Movement-related changes in population activity are carried mainly by the fields (firing rates); at 10-ms resolution there is no evidence that coupling strengthens during the reach, although moderate changes cannot be excluded.

**Status.** established / robust

**Tables copied here:** `energy_decomposition_tvfield_summary.csv`, `epochJ_summary.csv`, `j_rate_vs_cov_summary.csv`, `coupling_source_window_tests.csv`, `peak_timing_vs_control_all_variants.csv`, `dedup_effect.csv`, `late_epoch_consistency.csv`


### 01_coupling_source_checks.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/assets/figures/coupling_source_checks.png; PDF: ising_wt_protocol/07_presubmission_checks/assets/figures/coupling_source_checks.pdf
- **Axes:** (a) Change in coupling term per pair vs ms from peak velocity, split into the part predicted from rates and the trial-to-trial covariance part. (b) Held-out LLR (nats per 10-ms bin), Ising vs constant-rate independent. (c) Ising vs time-varying-rate independent.
- **Pattern:** Against constant rates Ising gains strongly during the reach; against time-varying rates the advantage dips below zero around acceleration and recovers afterwards.
- **Why it supports the claim:** The movement-time Ising advantage of the stationary model comes from reach-locked rate modulation (B10, B11).
- **Key numbers:** Ising worse during acceleration vs time-varying rates (2/21, p_FDR = 0.004); covariance part median 36% of the J rise (B10, B11).
- **PDF:** vector (copied from source)


### 02_rates_vs_couplings_heldout.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/rates_vs_couplings_heldout.png
- **Axes:** (a) Held-out gain over static independent model (nats/bin/neuron; median, IQR) vs time from 1-cm crossing for rates-only, couplings-only and both. (b) Rates-only minus couplings-only per session, baseline vs reach (+-200 ms), coloured by cohort. (c) Whole-trial gain per session for each model.
- **Pattern:** Rates dominate during the reach; couplings help at all times; the combined model is highest.
- **Why it supports the claim:** Shows that rate dynamics, not coupling changes, carry the reach.
- **Key numbers:** Rates-only wins in the reach in 11/13 sessions; the figure states p = 0.15 (the brief gives no p for 11/13). 'Time-varying Ising beats time-varying independent, 12/13, p = 0.0005' is stated in HANDOFF_CLAUDE_CODE.md section 7 and matches tvfield_phase_summary.csv (exact LL, N <= 20: 13 sessions, p_FDR = 0.0005); it is not drawn in this figure. Only 13 sessions are in this analysis.
- **PDF:** rasterized from PNG (2038x925 px at 300 dpi)


### 03_energy_decomposition_by_phase_STATIONARY.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/energy_decomposition_by_phase.png
- **Axes:** (a) Change in energy vs pre-reach baseline vs samples (10 ms) from peak velocity: total, coupling contribution, field contribution, STATIONARY model. (b) Per-session phase-mean contributions (accel., decel., post).
- **Pattern:** In the stationary model the energy drop is in the coupling term.
- **Why it supports the claim:** Shown ONLY with 03_tvfield_vs_independent panel c: with time-varying fields the drop moves to the field term. The coupling-carried drop is a property of the stationary model (B2, model-dependent).
- **Key numbers:** Stationary coupling: accel. p = 1e-5, decel. 1e-4, post 7e-4; field p >= 0.84. Time-varying model: field term 20/21, p_FDR = 2.3e-5 at acceleration (energy_decomposition_tvfield_summary.csv); in the time-varying model the coupling term changes about 20x less than the field term (about -0.01 vs -0.17 per neuron; RESULTS_LOG). This is NOT a comparison with the stationary model's coupling dip.
- **PDF:** rasterized from PNG (2184x887 px at 300 dpi)


### 04_J_timing_rise_decay_STATIONARY.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/J_timing_rise_decay.png
- **Axes:** (a) Coupling term z per reach (1249 reaches) sorted by J-peak time vs samples from peak velocity. (b) Session-mean J (z, +-SEM). (c) Density of single-reach J-peak times vs control window.
- **Pattern:** Stationary-model J is a broad elevation spanning the reach (peak z = 4.7 at +430 ms), not a timed peak; single-reach maxima cluster at peak velocity.
- **Why it supports the claim:** Context for B3/B7; stationary model, so the rise is largely rate-driven (B10, B11).
- **Key numbers:** 12.4% of J maxima within +-50 ms vs 6.4% in control, p < 1e-14 (B7).
- **PDF:** rasterized from PNG (2165x2111 px at 300 dpi)


### 05_peak_timing_vs_control.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/assets/figures/peak_timing_vs_control.png; PDF: ising_wt_protocol/07_presubmission_checks/assets/figures/peak_timing_vs_control.pdf
- **Axes:** (a) Per-session peak lag, real minus control window (ms), for energy, J, h, firing rate. (b) Median lag across 6 window/exclusion variants.
- **Pattern:** Only energy maxima deviate (earlier), and only in 2 of 6 variants.
- **Why it supports the claim:** Timing of J, h and rate maxima matches chance (B12); supersedes earlier timing claims (see X_).
- **Key numbers:** Energy p_FDR = 0.02; J, h, FR p_FDR = 0.39.
- **PDF:** vector (copied from source)


### 06_flat_FR_vs_J_modulation.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/flat_FR_vs_J_modulation.png
- **Axes:** Coupling-term movement modulation (peak |z|) vs firing-rate movement modulation (peak |z|), log-log, 63 conditions, colour = Ising advantage in P(K).
- **Pattern:** J and rate modulation move together; J modulation without rate modulation is rare.
- **Why it supports the claim:** B9: coupling changes almost always come with rate changes.
- **Key numbers:** rho = 0.74, n = 63; one clear example 210511 stim 1.
- **PDF:** rasterized from PNG (1487x1228 px at 300 dpi)


### 07_example_flatFR_210511_stim1.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/example_flatFR_210511_stim1.png
- **Axes:** Example session 210511 stim 1: kinematics, firing rate and Ising terms vs time from peak velocity (stationary model).
- **Pattern:** The clearest case of J modulation with a nearly flat mean rate.
- **Why it supports the claim:** Single example for B9; note 210511 contains a neuron firing in 99% of bins (see 1/02 panel b).
- **Key numbers:** Example only.
- **PDF:** rasterized from PNG (1303x1672 px at 200 dpi)


### 08_late_epoch_consistency_SUGGESTIVE.png / .pdf
- **Source:** ising_wt_protocol/07_presubmission_checks/timevarying_field/late_epoch_consistency.png; PDF: ising_wt_protocol/07_presubmission_checks/timevarying_field/late_epoch_consistency.pdf
- **Axes:** (a) Late Ising advantage, odd vs even reaches per session. (b) Late minus baseline gain (nats/bin), simultaneous vs 200-ms-lagged, by cohort. (c) Held-out gain from late-specific couplings.
- **Pattern:** Late (+0.5 to +1.5 s) increase only in the 2022 cohort; lagged control also rises; epoch-specific J does not help.
- **Why it supports the claim:** SUGGESTIVE ONLY (log section C). Shown so the late epoch is not over-read.
- **Key numbers:** 13/21, p_FDR = 0.023, carried by 2022 (6/6); 2021 7/15.
- **PDF:** vector (copied from source)


### 09_energy_absolute_with_firing_rate.png / .pdf
- **Source:** evidence/figures/3_rates_vs_coupling_over_reach/energy_absolute_with_firing_rate (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/3_rates_vs_coupling_over_reach/energy_absolute_with_firing_rate.pdf
- **Axes:** Absolute energy per neuron (not baseline-subtracted) vs ms from peak velocity, median and IQR over 21 sessions: total, field term and coupling term, for the time-varying-field model (left, with population firing rate on the right axis) and the stationary model (right). Blue/red = acceleration/deceleration.
- **Pattern:** With time-varying fields, energy drops sharply just before peak velocity and the drop is in the field term; the coupling term barely moves. The stationary model shows almost no reach-locked structure.
- **Why it supports the claim:** Shows the raw levels behind the change-from-baseline plots: the reach is carried by the fields (rates), with a nearly constant coupling contribution.
- **Key numbers:** M1 field term -0.15 at baseline to -0.335 (minimum at -50 ms); total -0.23 to -0.44 (-30 ms); coupling term about -0.04 to -0.08 with no comparable dip. Population spike probability per 10-ms bin 0.60 at baseline, minimum 0.52 at +120 ms. Exploratory (pseudo-likelihood fits).
- **PDF:** vector (copied from source)


## 4_polarization_at_peak_acceleration

**Claim.** The time-varying field term dips at peak acceleration far more precisely than the population-average rate, and this reflects population polarization (cells moving away from 50/50) that a model-free index captures equally well.

**Status.** robust (E12 preregistered; not unique to the Ising model)

**Tables copied here:** `primary_primary_family.csv`, `primary_signal_summary.csv`, `primary_descriptive_contrasts.csv`, `primary_secondary_vs_control.csv`, `E12_RESULTS.md`, `e10_tests.csv`, `fielddip_shuffle_controls.csv`, `burster_pauser_summary.csv`, `burster_pauser_by_condition.csv`


### 01_E12_T1_T2_by_signal.png / .pdf
- **Source:** ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/T1_T2_by_signal.png; PDF: ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/T1_T2_by_signal.pdf
- **Axes:** Left (T1): fraction of conditions whose trough falls in [-200, 0] ms before peak velocity, real (colour) vs rest control (grey), per signal (FT field term, FT0 uncoupled field term, POLb polarization index, dFR, FR mean rate, TRAJ PCA speed, DEC supervised decoder). Right (T2): |extremum - peak acceleration| (ms), session medians.
- **Pattern:** FT, FT0 and POLb hit far above rest controls and are close to peak acceleration; FR is late and imprecise; DEC is best but supervised.
- **Why it supports the claim:** Preregistered E12 test of timing (B13).
- **Key numbers:** Hit rate FT 0.65 vs 0.14 rest; FR 0.30. Median error FT 50 ms, FT0 70, POLb 70, FR 330; FT vs FR -208 ms, p_BH = 0.024; hit-rate difference p_BH = 0.078 (not FDR significant).
- **PDF:** vector (copied from source)


### 02_E12_signals_aligned.png / .pdf
- **Source:** ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/signals_aligned.png; PDF: ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/signals_aligned.pdf
- **Axes:** Condition-mean signals (baseline z, or decoder score) vs ms from peak acceleration, 63 conditions, mean +- s.e.m.
- **Pattern:** FT, FTu, FT0 and POLb all trough near peak acceleration; FR falls later and broadly; dFR rises.
- **Why it supports the claim:** Shows the shapes behind T1/T2.
- **Key numbers:** Do not read the trough DEPTH as evidence (withdrawn; basis inflation, E12 R6-R7).
- **PDF:** vector (copied from source)


### 03_E12_primary_contrasts_forest.png / .pdf
- **Source:** ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/primary_contrasts_forest.png; PDF: ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/primary_contrasts_forest.pdf
- **Axes:** FT minus each other signal with 95% bootstrap CIs: T1 hit fraction (left) and T2 ms (right); grey = equivalence margins.
- **Pattern:** FT beats FR clearly on T2; FT vs FT0 and FT vs POLb straddle zero.
- **Why it supports the claim:** The dip is not unique to the coupled model: no detectable advantage over FT0 or the model-free index (equivalence not shown).
- **Key numbers:** FT vs FT0: T1 +0.02 (CI -0.10 to +0.15), T2 -20 ms (CI -59 to +12). FT vs POLb: T2 -26 ms (CI -67 to +6). Trace correlations (r = 0.98 field term vs polarization index; r = -0.07 vs mean rate) are reproduced in 05_field_term_vs_polarization_vs_rate below (not part of E12 RESULTS.md).
- **PDF:** vector (copied from source)


### 04_field_vs_firing_rate_polarization_CAVEAT.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/fig5c_field_vs_firing_rate.png
- **Axes:** (a) Energy change per neuron (field and coupling terms) and population firing-rate change vs time from peak velocity. (b) Field term and population rate in baseline SD vs time from peak velocity. (c) Reach-window spike probability vs baseline spike probability, 1092 neurons, coloured toward/away from 0.5.
- **Pattern:** Field term troughs at peak acceleration; 60% of neurons move away from p = 0.5 during the reach.
- **Why it supports the claim:** Panel c illustrates polarization. CAVEAT: panel a's title ('average firing rate barely moves') and panel b's baseline-SD depth are WITHDRAWN claims (log D: the mean rate does dip, and the depth is basis-inflated). Cite panel c only.
- **Key numbers:** 60% of 1092 neurons away from 0.5. E10 H1: change in p(1-p) negative in 19/21 sessions (median -0.027, p = 7e-6; e10_tests.csv H1_dv).
- **PDF:** rasterized from PNG (2245x948 px at 300 dpi)


### 05_field_term_vs_polarization_vs_rate.png / .pdf
- **Source:** evidence/figures/4_polarization_at_peak_acceleration/field_term_vs_polarization_vs_rate (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/4_polarization_at_peak_acceleration/field_term_vs_polarization_vs_rate.pdf
- **Axes:** (a) Baseline-z traces vs ms from peak acceleration, 63 conditions, mean +- s.e.m.: Ising field term, field term without coupling, polarization index (no model), average firing rate. (b) Per-condition correlation of each trace with the Ising field term (-600 to +1000 ms).
- **Pattern:** Field term, uncoupled field term and polarization index are almost the same curve; the mean rate has a different shape.
- **Why it supports the claim:** The field-term dip is a polarization signal, not a mean-rate signal, and it does not need the coupled model.
- **Key numbers:** Median r with the Ising field term: uncoupled field term 0.98, polarization index 0.98, mean rate -0.07. EXPLORATORY.
- **PDF:** vector (copied from source)


### 06_fielddip_shuffle_controls.png / .pdf
- **Source:** evidence/figures/4_polarization_at_peak_acceleration/fielddip_shuffle_controls (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/4_polarization_at_peak_acceleration/fielddip_shuffle_controls.pdf
- **Axes:** (a) Fraction of field-term dips (E12 search window) in the 200 ms before peak velocity and (b) their distance from peak acceleration, for real data and three shuffles (5 each per block); one dot per session.
- **Pattern:** Shifting each cell in time destroys the timing; shuffling reaches across cells (each cell keeps its average time course) does not; using another session's landmarks barely changes anything.
- **Why it supports the claim:** The dip timing comes from each cell's reach-locked time course, not from reach-by-reach co-variation between cells. The landmark shuffle is a weak control because peak acceleration sits at nearly the same bin in every session.
- **Key numbers:** Hit rate real 0.65, trial shuffle 0.67 (p = 0.60), per-cell time shift 0.15 (p = 1e-4), landmark shuffle 0.61 (p = 0.40). Median distance 100 / 87 / 319 / 101 ms. Burster/pauser table: the rate extremum matching the cell type is not detectably closer to the field-term trough than the other extremum (280 vs 420 ms, p = 0.14; 24 burster and 39 pauser conditions). EXPLORATORY.
- **PDF:** vector (copied from source)


## 5_fixed_coupling_influence_E10

**Claim.** Preregistered E10: with J fixed, coupling influence (multi-information per neuron) does NOT rise when cells polarize during the reach; the overall hypothesis is not supported.

**Status.** preregistered-null (REFUTED / NOT SUPPORTED overall)

**Tables copied here:** `e10_tests.csv`, `e10_sessions.csv`, `e10_convergence_by_cohort.csv`, `e10_exact_null_n19.csv`, `e10_warnings.txt`, `step1_summary.csv`


### 01_e10_h2_h4.png / .pdf
- **Source:** ising_wt_e10/10_operating_point_h2_h4/assets/figures/e10_h2_h4.png
- **Axes:** Left (H4a readout): session dI/N (bits/neuron) vs session mean change in p(1-p) (reach - baseline); red = 2022 cohort, blue = 2021. Middle (H3a): empirical time-locked dcov/N vs fixed-J predicted dchi_J/N (63 conditions). Right (H3b): histogram of session-mean coupling scale kappa-hat.
- **Pattern:** dI/N clusters near zero and is mostly slightly negative (2021); the 2022 sessions scatter positive. Predicted and empirical covariance changes correlate weakly; kappa-hat is mostly positive but wide.
- **Why it supports the claim:** The preregistered H2 test fails, H3a fails on sign agreement, so the fixed-coupling operating-point account is not supported.
- **Key numbers:** H2 dI/N median -0.0011, 15/21 negative, p = 0.29; 2021 alone -0.0014, 13/15 negative, p = 0.015; 2022 +0.007 (6, p = 0.44). H3a rho = 0.34, sign agreement 13/21 (p = 0.38). H3b kappa median +0.19, 13/21, p = 0.055 (verdict: no change; but 2022 alone 6/6, p = 0.031). H4a rho = 0.70 (near-mechanical). Convergence: 2022 real fits not converged 54/99 tasks; 2021 19/180.
- **PDF:** rasterized from PNG (1950x600 px at 150 dpi)


### 02_e10_step1_synthetic_amendments.png / .pdf
- **Source:** ising_wt_e10/10_operating_point_h2_h4/assets/figures/step1_synthetic_amendments.png
- **Axes:** SYNTHETIC. Left: kappa fit in scenarios A (truth 0) and C (truth 0.6) by N and reaches. Right: rho_real - rho_null in scenarios D (truth 0) and E (< 0).
- **Pattern:** kappa recovers 0 and 0.6; rho_real - rho_null cannot detect scenario E.
- **Why it supports the claim:** Method validation; shows why the neuron-level test H4b was demoted to exploratory.
- **Key numbers:** Synthetic only.
- **PDF:** rasterized from PNG (1500x600 px at 150 dpi)


### 03_h0_synthetic_validation.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/h0_synthetic_validation.png
- **Axes:** SYNTHETIC. (a) Estimated change from the fixed-J model vs true change in multi-information (bits/neuron), scenarios A-E. (b) Change in across-reach covariance per neuron (data points vs fixed-J prediction bars). (c) Neuron-level rho (closer to 0.5 vs change in influence).
- **Pattern:** Rate-driven changes are recovered on the identity line; a true J change (C) is flagged as unexplained; the neuron-level test is biased without a null.
- **Why it supports the claim:** Validation of the E10 measures before the real-data test.
- **Key numbers:** Synthetic only. Labelled h0_synthetic_validation in 09_claude_science_handoff (panel a is redrawn with per-scenario counts as 1/05).
- **PDF:** rasterized from PNG (1988x925 px at 300 dpi)


## S1_stimulation_E11

**Claim.** Closed-loop optogenetic stimulation: the change in coupling scale kappa in the 60 ms after laser onset is inconclusive (leans negative, not significant after FDR, not equivalent to zero), and multi-information does not change.

**Status.** inconclusive (preregistered, refit under amendments A3-A5)

**Tables copied here:** `stats_summary.json`, `session_table.csv`, `E11_RESULTS.md`


### 01_E11_S1_dkappa_per_session.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig1_S1_dkappa_per_session.png
- **Axes:** Raw dkappa = kappa_pulse - kappa_control per session (21 sessions, coloured by cohort); line = Hodges-Lehmann estimate, dark band = 90% CI, grey = +-0.3 equivalence band.
- **Pattern:** 14/21 sessions negative; CI excludes 0 but extends beyond -0.3.
- **Why it supports the claim:** Primary S1 result: outcome (c) inconclusive.
- **Key numbers:** HL -0.266, 90% CI [-0.469, -0.022]; Wilcoxon p = 0.065, p_FDR = 0.13.
- **PDF:** rasterized from PNG (960x840 px at 150 dpi)


### 02_E11_S1_sensitivity.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig2_S1_sensitivity.png
- **Axes:** Hodges-Lehmann dkappa with 90% CI for the primary analysis and sensitivity variants (control window, onset rounding, exclusions, cohorts).
- **Pattern:** All variants lean negative except the 2022 cohort; the fixed control is near zero.
- **Why it supports the claim:** The lean depends on the control window and is carried by 2021.
- **Key numbers:** 2021 -0.44 (n = 15, p = 0.026 nominal); 2022 +0.17 (n = 6, p = 0.56); fixed control -0.09, p = 0.41.
- **PDF:** rasterized from PNG (990x630 px at 150 dpi)


### 03_E11_S2_multiinformation.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig3_S2_thermo_per_session.png
- **Axes:** Per-session pulse - control change in multi-information / N (bits; left) and chi_J / N (right).
- **Pattern:** Centred on zero; outliers in 2022 (220516).
- **Why it supports the claim:** S2: no change in coupling influence in the pulse window.
- **Key numbers:** dI/N HL -0.0002, p = 0.79; dchi_J/N HL -0.007, p = 0.68.
- **PDF:** rasterized from PNG (1200x540 px at 150 dpi)


## S2_shuffle_controls_energy_timeseries

**Claim.** The reach-locked field/energy dip and the rate change are destroyed by moving cells in time but preserved by the trial shuffle; the coupling level is destroyed by the trial shuffle and per-neuron shift but preserved by the whole-population shift.

**Status.** exploratory (shuffle controls requested by the PI; not preregistered)

**Tables copied here:** `shuffle_test_summary.csv`, `shuffle_test_by_session.csv`


### 01_energy_timeseries_vs_shuffles.png / .pdf
- **Source:** evidence/figures/S2_shuffle_controls_energy_timeseries/energy_timeseries_vs_shuffles (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S2_shuffle_controls_energy_timeseries/energy_timeseries_vs_shuffles.pdf
- **Axes:** Field term, coupling term, total energy (per neuron) and population firing rate vs ms from peak velocity: real data and the time-varying-field model refitted on 20 shuffles of each type per block (trial shuffle = reach identity permuted per cell; per-neuron time shift; whole-population time shift). Median over 21 sessions.
- **Pattern:** Time shifts flatten the dip and the rate change; the trial shuffle keeps the dip (each cell's average time course is intact) but drives the coupling term to ~0; the population shift keeps the coupling level.
- **Why it supports the claim:** Answers the PI's shuffle question: the dip is a reach-locked property of each cell's time course; the couplings come from same-bin co-firing across cells within a reach, which the trial shuffle breaks.
- **Key numbers:** Sessions beyond all 20 shuffles (dip = mean -300..+100 ms minus baseline -1000..-400 ms): field dip 18/21 per-neuron shift, 19/21 population shift, 6/21 trial shuffle (ns, p = 0.24); total dip 19, 19, 7/21 (ns); FR change 19/21 and 16/21 (time shifts); coupling level 21/21 trial, 19/21 per-neuron shift, 6/21 population shift (ns). Wilcoxon p < 1e-5 for every non-ns row.
- **PDF:** vector (copied from source)


## S3_bin_size_1ms

**Claim.** Refitting at 1-ms bins keeps the field-term dip (narrower, about 45 ms later, near peak velocity) but the coupling-term dip disappears.

**Status.** exploratory

**Tables copied here:** `gate_report.csv`


### 01_compare_1ms_vs_10ms_scaled.png / .pdf
- **Source:** evidence/figures/S3_bin_size_1ms/compare_1ms_vs_10ms_scaled (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S3_bin_size_1ms/compare_1ms_vs_10ms_scaled.pdf
- **Axes:** Field term, coupling term and population firing rate, change from baseline scaled to each curve's maximum, 1-ms fit (10-ms running mean) vs 10-ms fit, vs ms from peak velocity; median over 21 sessions. Dashed = median peak acceleration.
- **Pattern:** The field-term dip is present at both bin sizes; at 1 ms its minimum is about 45 ms later. The coupling-term dip seen at 10 ms is absent at 1 ms.
- **Why it supports the claim:** The field-term result does not depend on the 10-ms binning; the coupling-term dip may.
- **Key numbers:** Minimum of field term: -50 ms (10-ms bins) vs -4 ms (1-ms bins), CI wide. All gates passed (1-ms spikes re-binned reproduce the 10-ms data exactly); 2,184 fits converged. J at 1 ms vs 10 ms r = 0.61. Data flag: 210511 unit 11 looks duplicated (74% of active 1-ms bins have >= 2 spikes).
- **PDF:** vector (copied from source)


### 02_energy_absolute_1ms.png / .pdf
- **Source:** evidence/figures/S3_bin_size_1ms/energy_absolute_1ms (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S3_bin_size_1ms/energy_absolute_1ms.pdf
- **Axes:** Absolute energy terms per neuron per 1-ms bin (total, field, coupling) and firing rate (right axis), median over 21 sessions, 10-ms running mean.
- **Pattern:** Same layout as 3/09 at 1 ms. Per-ms firing rises during the reach while P(>= 1 spike per 10 ms) falls, i.e. spikes cluster.
- **Why it supports the claim:** Context for 01.
- **Key numbers:** Energy scales are not comparable across bin sizes.
- **PDF:** vector (copied from source)


## S4_entropy_and_learning_exploratory

**Claim.** Entropy does not differ between baseline, stimulation and washout blocks; scored against a baseline-fitted model, stimulation reaches depart progressively and washout reaches stay away, in firing rates rather than coupling. Recording drift cannot be excluded.

**Status.** exploratory (confounded with time in session)

**Tables copied here:** `baseline_ref_stats.csv`


### 01_entropy_by_block.png / .pdf
- **Source:** evidence/figures/S4_entropy_and_learning_exploratory/entropy_by_block (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S4_entropy_and_learning_exploratory/entropy_by_block.pdf
- **Axes:** Entropy, independent entropy and multi-information per neuron (bits per 10-ms bin) vs ms from peak velocity, separate fits per block (stim 0 baseline, 1 stimulation, 2 washout); median and IQR over 21 sessions.
- **Pattern:** The three blocks overlap.
- **Why it supports the claim:** No block-level change in the model's entropy or in coupling-carried information.
- **Key numbers:** Session-level Wilcoxon (window -1000..+2000 ms): all block contrasts p >= 0.20.
- **PDF:** vector (copied from source)


### 02_baseline_referenced_surprise.png / .pdf
- **Source:** evidence/figures/S4_entropy_and_learning_exploratory/baseline_referenced_surprise (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S4_entropy_and_learning_exploratory/baseline_referenced_surprise.pdf
- **Axes:** Top: surprise of each reach under a model fitted on that session's baseline reaches, minus the held-out baseline mean (bits/neuron/bin), vs reach number across baseline, stimulation, washout. Bottom: part due to coupling (full-model change minus rates-only change). Median and IQR over sessions.
- **Pattern:** Surprise climbs through the stimulation block and stays high in washout; the coupling part stays near 0.
- **Why it supports the claim:** Suggests the population code moves away from baseline during stimulation without recovering, carried by rates. Could be learning, or simply time in session.
- **Key numbers:** Stim vs baseline 17/21 above, p = 0.0004; within-stimulation trend 18/21 positive, p = 0.006; washout vs baseline 19/21, p < 1e-4, no recovery trend (p = 0.92); coupling part p = 0.52 (stim), 0.95 (washout). CAVEAT: drift.
- **PDF:** vector (copied from source)


## S5_per_reach_locking_exploratory

**Claim.** Reach by reach, the dip is only weakly locked to that reach's peak acceleration; re-aligning to each reach's own peak acceleration does not sharpen the average dip beyond shuffled alignment.

**Status.** exploratory (low power)

**Tables copied here:** `per_reach_correlation_summary.csv`, `realign_summary.csv`


### 01_realign_own_vs_shuffled.png / .pdf
- **Source:** evidence/figures/S5_per_reach_locking_exploratory/realign_own_vs_shuffled (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S5_per_reach_locking_exploratory/realign_own_vs_shuffled.pdf
- **Axes:** Field term and total energy per reach, aligned to each reach's own peak acceleration (black) vs a shuffled reach's peak acceleration (red, 1,000 shuffles), averaged; change from baseline, median over sessions.
- **Pattern:** The two curves are nearly identical.
- **Why it supports the claim:** Per-reach jitter of peak acceleration is small relative to the width of the dip, so this test has little power either way.
- **Key numbers:** Depth own -0.160 vs shuffled -0.161 (field, 14/21 deeper, p = 0.37); total -0.173 vs -0.172 (p = 0.19). Per-reach rho(dip time, peak-acc time) field 0.09 vs null 0.00, 13/21, p = 0.04; distance 113 vs 119 ms, p = 0.12. Peak-acceleration jitter SD 54 ms vs a dip ~300 ms wide.
- **PDF:** vector (copied from source)


## S6_model_comparison_M0_vs_M1

**Claim.** With exact likelihood on held-out reaches, adding couplings (M1) to the time-varying rates (M0) improves prediction in every session over the whole trial, and also at rest and during the reach.

**Status.** exploratory check (supports established A7)

**Tables copied here:** `M1_vs_M0_exact.csv`, `tvfield_phase_summary.csv`


### 01_M1_vs_M0_exact.png / .pdf
- **Source:** evidence/figures/S6_model_comparison_M0_vs_M1/M1_vs_M0_exact (regenerated 2026-10-27 by evidence/src/); PDF: ising_wt_evidence/evidence/src/package/../../figures/S6_model_comparison_M0_vs_M1/M1_vs_M0_exact.pdf
- **Axes:** Held-out gain of couplings (M0 minus M1 surprise, bits per neuron per 10-ms bin) per session, for rest (-1000..-400 ms), reach (+-200 ms) and whole trial (-1000..+2000 ms) around the 1-cm crossing. 5-fold CV over reaches, exact likelihood (fixed 16-cell subset for larger sessions).
- **Pattern:** Every session is above 0 over the whole trial; nearly all at rest and during the reach.
- **Why it supports the claim:** Exact-likelihood confirmation that couplings add predictive power beyond time-varying rates, including at rest (A7).
- **Key numbers:** Whole trial 21/21 sessions, 63/63 conditions, median 0.0115 bits/neuron/bin (median relative gain 1.6% of M0 surprise), p = 9.5e-7; rest 19/21, 51/63, p = 1.8e-5; reach 20/21, 59/63, p = 1.9e-6.
- **PDF:** vector (copied from source)


## X_withdrawn_or_superseded

**Claim.** Older figures whose claims were withdrawn, retracted or superseded; kept only for provenance.

**Status.** withdrawn / superseded (do NOT cite as evidence)

**Tables copied here:** `RETRACTED_E11_stats_summary_run1_0to700.json`, `E11_RESULTS_run1_SUPERSEDED.md`


### 01_WITHDRAWN_prelim_field_vs_rate_trough_E09.png / .pdf
- **Source:** ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/prelim_field_vs_rate_trough.png; PDF: ising_wt_protocol/12_field_vs_rate_indicator/assets/figures/prelim_field_vs_rate_trough.pdf
- **Axes:** (a) Trough time histograms, (b) % troughs in 200 ms before peak velocity, (c) trough depth / rest depth per session.
- **Pattern:** Exploratory E09 version of the E12 test.
- **Why it supports the claim:** SUPERSEDED by E12 T1_T2_by_signal. Panel c depth ratio is WITHDRAWN (basis inflation, E12 R7).
- **Key numbers:** -
- **PDF:** vector (copied from source)


### 02_WITHDRAWN_j_peak_timing_vs_velocity_peak.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/j_peak_timing_vs_velocity_peak.png
- **Axes:** J-peak time minus peak-velocity time (samples) for 37 collective reaches.
- **Pattern:** Peaks split across the acc./dec. boundary.
- **Why it supports the claim:** WITHDRAWN: 'J peaks after peak acceleration' / 'J peaks in deceleration' (window-length artefact; log D).
- **Key numbers:** -
- **PDF:** rasterized from PNG (1865x1279 px at 300 dpi)


### 03_WITHDRAWN_pilot_criticality_proxy.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/pilot_criticality_proxy.png
- **Axes:** Pilot session 210421: kinematics, Var(energy), Var(J) and Var(h) vs time bin.
- **Pattern:** Pilot variance peaks.
- **Why it supports the claim:** DID NOT REPLICATE: criticality proxy Var(J) peak (log D).
- **Key numbers:** -
- **PDF:** rasterized from PNG (1170x1040 px at 130 dpi)


### 04_WITHDRAWN_decision_tree_generalization.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/decision_tree_generalization.png
- **Axes:** Training vs leave-one-session-out accuracy for decision-tree variants; per-session accuracy.
- **Pattern:** CV accuracy at or below majority baseline.
- **Why it supports the claim:** ABANDONED: decision tree predicting J peaks (log D).
- **Key numbers:** -
- **PDF:** rasterized from PNG (2009x1590 px at 200 dpi)


### 05_PARTLY_WITHDRAWN_phase_timing_summary.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/phase_timing_summary.png
- **Axes:** (a) z vs baseline of coupling, field, rate, energy; (b) dominant-extremum timing; (c) movement-modulation depth; (d) within-phase correlation with acceleration.
- **Pattern:** -
- **Why it supports the claim:** Panel b is DROPPED (superseded by 3/05 peak_timing_vs_control). Panels a, c, d still support B4/B6.
- **Key numbers:** FR wins 15-18/21 (B6).
- **PDF:** rasterized from PNG (2141x1594 px at 300 dpi)


### 06_SUPERSEDED_sensitivity_robustness.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/sensitivity_robustness.png
- **Axes:** Signed -log10 p for many window/exclusion variants.
- **Pattern:** -
- **Why it supports the claim:** Peak-timing rows SUPERSEDED by peak_timing_vs_control; 'coupling term drives the energy drop' holds only for the stationary model.
- **Key numbers:** -
- **PDF:** rasterized from PNG (2372x2510 px at 300 dpi)


### 07_SUPERSEDED_stim_block_reach_trajectories.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/stim_block_reach_trajectories.png
- **Axes:** Change vs last 5 baseline reaches (baseline SD) of coupling term, field term and firing rate across stimulation and washout blocks.
- **Pattern:** Coupling term above baseline in the stim block (uncorrected).
- **Why it supports the claim:** PRELIMINARY per-block fits; none survive FDR; SUPERSEDED by E11 (S1).
- **Key numbers:** -
- **PDF:** rasterized from PNG (2177x1079 px at 300 dpi)


### 08_RETRACTED_E11_run1_fig1.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig1_run1_0to700_SUPERSEDED.png
- **Axes:** As S1/01 (run 1).
- **Pattern:** -
- **Why it supports the claim:** RETRACTED: run 1 fitted bins 0-699 instead of 300-559 and used a misaligned null.
- **Key numbers:** -
- **PDF:** rasterized from PNG (960x840 px at 150 dpi)


### 09_RETRACTED_E11_run1_fig2.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig2_run1_0to700_SUPERSEDED.png
- **Axes:** As S1/02 (run 1).
- **Pattern:** -
- **Why it supports the claim:** RETRACTED (run 1).
- **Key numbers:** -
- **PDF:** rasterized from PNG (990x630 px at 150 dpi)


### 10_RETRACTED_E11_run1_fig3.png / .pdf
- **Source:** ising_wt_protocol: git show origin/exp/11_stimulation_kappa:11_stimulation_kappa/assets/figures/fig3_run1_0to700_SUPERSEDED.png
- **Axes:** As S1/03 (run 1).
- **Pattern:** -
- **Why it supports the claim:** RETRACTED (run 1).
- **Key numbers:** -
- **PDF:** rasterized from PNG (1200x540 px at 150 dpi)


### 11_SUPERSEDED_Sept30_findings_Figure1.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 2)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 1: The outward reach is brief.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 12_SUPERSEDED_Sept30_findings_Figure2.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 3)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 2: Population activity is more collective than independent.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 13_SUPERSEDED_Sept30_findings_Figure3.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 4)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 3: Fitted couplings are not a fitting artefact.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 14_SUPERSEDED_Sept30_findings_Figure4.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 5)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 4: Validation of the time-varying-field model.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 15_SUPERSEDED_Sept30_findings_Figure5.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 6)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 5: Couplings add prediction beyond shared rate changes.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 16_SUPERSEDED_Sept30_findings_Figure6.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 7)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 6: Where the stationary model's movement-time coupling signal comes from.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 17_SUPERSEDED_Sept30_findings_Figure7.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 8)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 7: Energy decomposition in the stationary model.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 18_SUPERSEDED_Sept30_findings_Figure8.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 9)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 8: J is a broad elevation around the reach, not a timed peak.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 19_SUPERSEDED_Sept30_findings_Figure9.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 10)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 9: Peak timing of J, h and firing rate matches chance.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 20_SUPERSEDED_Sept30_findings_Figure10.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 11)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 10: Firing rate is the most movement-modulated signal.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 21_SUPERSEDED_Sept30_findings_Figure11.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 12)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 11: J changes without firing-rate changes are rare.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 22_SUPERSEDED_Sept30_findings_Figure12.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 13)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 12: The clearest dissociation: 210511 stim 1.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 23_SUPERSEDED_Sept30_findings_Figure13.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 14)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 13: Suggestive: a late post-reach coupling increase.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 24_SUPERSEDED_Sept30_findings_Figure14.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 15)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 14: What distinguishes the sessions with the strongest effects.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


### 25_SUPERSEDED_Sept30_findings_Figure15.png / .pdf
- **Source:** ising_wt_protocol/09_claude_science_handoff/assets/figures/ISING_FINDINGS_FIGURES.pdf (page 16)
- **Axes:** Page of the 2026-09-30 findings PDF: figure plus its caption.
- **Pattern:** -
- **Why it supports the claim:** SUPERSEDED by the 2026-10-13 RESULTS_LOG and the folders above (e.g. stationary-model energy decomposition is model-dependent; the field-dip depth and 'flat mean rate' claims are withdrawn).
- **Key numbers:** Sept-30 Figure 15: Preliminary: changes across baseline, stimulation and washout.
- **PDF:** vector (page split from source PDF); PNG rendered at 200 dpi


## Corrections and verification status (all numbers checked 2026-10-27)
1. **B-spline count:** 67 (code). The `tvfield_walkthrough.png` title says 66; the title is off by one.
2. **"Sign correct 40/40":** verified. It is scenarios A and B of the H0 synthetic test (20 + 20); see 1/05 for every scenario.
3. **"Pseudo-likelihood vs exact gain rho = 0.999":** NOT found in any saved table. Do not cite; the saved result is "exact and PL agree in 38/39 conditions".
4. **Held-out coupling gain:** 21/21 sessions, 63/63 above the shuffle null 95th percentile (A7), median +0.26 nats/bin. At rest (baseline phase): 20/21.
5. **200-ms lagged control:** recovers about 1/5 (0.051 vs 0.259 nats/bin, tvfield_phase_summary) or about 1/4 (0.05 vs 0.18, RESULTS_LOG). These are two different summaries; say "a small fraction (1/5 to 1/4)".
6. **"20x smaller":** compares the time-varying model's *coupling term* with its *field term* (about -0.01 vs -0.17 per neuron). It is not a comparison with the stationary model.
7. **Rates vs couplings (13 sessions):** rates-only beats couplings-only during the reach in 11/13 sessions, p = 0.15 (not significant). Time-varying Ising beats time-varying independent 12/13, p = 0.0005: HANDOFF section 7 and tvfield_phase_summary.csv (exact LL, p_FDR = 0.0005).
8. **E12 r = 0.98 / -0.07:** reproduced (4/05). These are exploratory trace correlations, not E12 preregistered outputs.
9. **E10 H3b:** the verdict is "no J change", but kappa median +0.19 (13/21, p = 0.055) and the 2022 cohort alone is 6/6 positive (p = 0.031).
10. **E10 convergence:** 54/99 non-converged real-fit tasks in 2022, 19/180 in 2021.
11. **P(K) 21/26:** reproduced (2/04).
12. **S6 M1 vs M0:** reproduced (21/21, 63/63, median 0.0115 bits/neuron/bin, p = 9.5e-7; rest 19/21; reach 20/21). The relative gain is 1.6% as the median of per-condition ratios; the earlier "1.3%" used a different summary.
13. **Exploratory numbers reproduced from regenerated code:**
    - S2: 18-19/21 vs 6-7/21; coupling 21/21, 19/21, 6/21.
    - S4: 17/21 (p = 0.0004); trend 18/21 (p = 0.006); washout 19/21.
    - S5: rho 0.09, 13/21 (p = 0.04); re-alignment p = 0.37; jitter 54 ms.
14. **Verified exactly, unchanged:**
    - synthetic +0.107 / about 0 / about 0, with r_J = 0.978;
    - |J| 63/63;
    - field 20/21, p_FDR = 2e-5;
    - A8 -15% (CI -28 to +93%);
    - E12: 65% vs 14%; 50 vs 330 ms; p_BH = 0.024;
    - E10: H2 -0.001, 15/21, p = 0.29; 2021 13/15, p = 0.015; H3a 0.34 and 13/21; H4a 0.70; H1 19/21;
    - E11: -0.266 [-0.469, -0.022], p_FDR 0.13; S2 p = 0.79; 2021 -0.44; 2022 +0.17.

## Still not recoverable
- The curated `paper_figures_by_point/` panels (1a-4e) and the old `findings_png/` set: repository equivalents and the split Sept-30 PDF pages (X_) are used instead.
- Data flag (not a missing file): session 210511 unit 11 looks like a duplicated unit at 1 ms (74% of active 1-ms bins have >= 2 spikes). Check spike sorting before final analyses.

## Repository figures not included (available in `09_claude_science_handoff/assets/figures/`)
strong_session_profile, session_diagnostics, example_flatFR_210608_stim1, example_flatFR_210619_stim2, example_flatFR_220520_stim0, example_phase_timing_220517_stim1. fig2_collective, fig3_not_artefact, fig4_tvfield_validation, fig5_coupling_beyond_rate, fig10_FR_most_modulated are raster pages of the Sept-30 findings PDF and are covered by the split pages in X_.
