# E12 pre-registration (dated 2026-10-12, author: grad-researcher-3)

Frozen BEFORE any signal, statistic or fit is computed on the real spikes. Later changes appear only as dated amendments at the bottom.

## 0. Disclosure of what was already seen
- Exploratory look (2026-10-12, from `09_.../assets/tables/field_term_trough_timing.csv`, 63 conditions, n = 21 sessions): field trough within 200 ms before peak velocity in 60% vs 14% of rest controls (p = 0.0002); average rate 32% vs 12% (p = 0.13); depth relative to rest control 4.7x (field) vs 2.0x (rate), field higher in 16/21 sessions (p = 0.12). Median field-trough time 70 ms from peak acceleration vs 210 ms for the rate (exploratory, unverified provenance of that table's exclusions).
- Before writing this document the author looked only at file structure (keys, array shapes, N and R per condition, landmark columns), never at spike values, fits or signals computed from them. The HANDOFF states the M1 field trough sits at a median of -40 ms from peak velocity.
- The generating script of `field_term_trough_timing.csv` is lost. E12 recomputes everything from `spikes_v2.npz` with `exclude_k == 0`. The old table is used for nothing but the disclosure above.

## 1. Data and conventions
- `spikes_v2.npz`, 63 conditions (21 sessions x 3 stim blocks), reaches with `exclude_k == 0` only. X_k: (R, 700, N) binary, 10-ms bins, s = 2x - 1.
- Landmarks: `09_.../assets/tables/phase_landmarks.csv`. `zc` = `zero_cross` (peak velocity), `pa` = `peak_acc`. Times are in bins relative to the array index (bin = 10 ms).
- Mouse map (from `reextract_sessions.csv`): fChR2 {210421,210422,210423,210425,210514}, fChR5 {210511,210512,210515}, fChR4 {210606,210608,210614}, cChR1 {210619,210620,210622,210623}, DJC002 {220515,220517,220519}, DJC000 {220516,220518,220520}. Cohort = 2021 (15 sessions) vs 2022 (6).
- Every signal is a per-condition time series of 700 bins built from the reach-average, then smoothed with a Gaussian of sigma = 3 bins (30 ms), identical for all signals, then z-scored with its own baseline (bins zc-100 .. zc-40, i.e. -1000 to -400 ms; mean and SD over those bins). The sign convention below makes every signal's expected movement-related event an EXTREMUM in the stated direction.

## 2. Signals (all from the same reaches)
| id | definition | extremum sought |
|---|---|---|
| FT | field term of M1 (`tvfield_model.fit(X, B, "M1")`, B = `bspline_basis(zc)`, default penalties): FT(t) = -sum_i h_i(t) * mean_r s_i(r,t) | min (trough) |
| FT0 | same formula with fields from M0 (J = 0; h = time-varying independent) | min |
| FR | population mean spike probability, mean over neurons and reaches | min (trough). Sensitivity FR_abs: the extremum of larger |z| in either direction |
| dFR | mean over neurons of |p_i(t) - p_i,base| (p_i,base = baseline mean of cell i's PSTH) | max |
| POLa | sum_i |p_i(t) - p_i,base| | max |
| POLb | -sum_i m_i(t) atanh(m_i(t)), m = 2p - 1 from the smoothed PSTH, p clipped to [0.01, 0.99] | min |
| TRAJ | speed of the PCA trajectory: PSTH matrix (700 x N, each cell z-scored by its baseline), top k = 3 PCs (fit on all 700 bins), speed = norm of the central difference of PC scores | max. Sensitivity: k with 90% variance; distance from baseline centroid (max) |
| DEC | cross-validated ridge decoder (see 4) | max of held-out score |
POLa and dFR differ only by a factor N and are the same test. dFR is reported once, called dFR. POOL (Calame-style pooled, velocity-binned rate) needs per-reach kinematics that are not in `spikes_v2.npz`; it is run only if kinematics become available before real-data analysis, and otherwise reported as NOT RUN. GPFA is not run (no validated implementation in this environment; stated as a limitation, not as evidence of absence). dPCA reduces to PCA here because each condition has a single task condition (time only).

## 3. Extremum search and tests
- Search window: [pa - 50, pa + 100] bins (the window fixed in advance in the phase-timing report). Extremum = the single best bin (argmin or argmax per the table) on the smoothed, baseline-z-scored signal.
- Rest controls: the same search with all quantities (window and pa) shifted by -150, -200 and -250 bins (-1.5/-2/-2.5 s). Control hit rates and delays are averaged over the three shifts within a condition.
- Rest check (data-limited): the control windows are assumed to be at rest. This is verified with unscaled hand kinematics if available; otherwise it is reported as unverified and the cell-level stationarity of FR over the control windows is shown instead.
- Unit: session (n = 21); condition values averaged within session. Two-sided Wilcoxon signed-rank, BH-FDR within the family below.
- T1 (timing): hit = extremum bin in [zc - 20, zc] (200 ms before peak velocity). Per session: d1 = fraction of hits (3 conditions) minus control fraction. Per signal, test d1 > 0 (secondary family). Comparison between signals A and B: Wilcoxon on d1_A - d1_B.
- T2 (precision): per condition |extremum bin - pa|; per session median; controls use |extremum - shifted pa|. Comparison between signals: Wilcoxon on session medians (real windows).
- T3 (depth): |z| at the extremum divided by the mean control |z| at the control extrema, per condition, session median of the ratio; FT vs FR comparison on log ratios. Also reported for FT0, POLb, dFR, TRAJ as descriptive.
- Primary family (12 tests, one BH-FDR): T1: FT vs {FR, dFR, FT0, POLb, TRAJ}; T2: FT vs {FR, dFR, DEC, FT0, POLb, TRAJ}; T3: FT vs FR. POLa == dFR is not counted twice. T1 has 5, T2 has 6, T3 has 1, so the family size is 12.
- Secondary family (BH separately): each signal's T1 d1 > 0 and T3 ratio > 1 against its own control.
- U1 = FT vs FT0 (T1 and T2). U2 = FT vs POLb, dFR, TRAJ (T1 and T2). Both reported with 95% CIs: bootstrap over sessions (10,000 resamples) of the mean paired difference, plus the Hodges-Lehmann estimate.
- Cohort: every test also run within 2021 (n = 15) and 2022 (n = 6), and the cohort difference by Mann-Whitney. Mouse: session-level paired differences fit as `diff ~ 1 + (1 | mouse)` (statsmodels MixedLM, 6 mice), reporting the intercept and CI.
- Sensitivity (not in the family): (i) all reaches (exclusions ignored); (ii) sessions with N >= 5 neurons per condition (conditions with N < 5 dropped); (iii) smoothing sigma = 2 and 5 bins; (iv) FR_abs; (v) hit window [zc - 30, zc].

## 4. DEC
- Per condition, features = each cell's Gaussian-smoothed (sigma = 3) single-reach rate in each bin, z-scored over the condition. Target y(t) = exp(-(t - pa)^2 / (2 * 5^2)). Training/evaluation bins = the search window [pa - 50, pa + 100]. Control windows are disjoint from it (shift >= 150), and the decoder trained on the real window is applied to the shifted bins.
- Ridge penalty by nested CV (inner 3-fold over training reaches, grid 1e-2 .. 1e4 log-spaced). Outer folds = `fold_k` of the npz (5 folds over reaches, as in E07). The held-out decoder score is averaged across reaches per time bin; the DEC extremum = argmax of that average over the search window. Held-out R^2 of the decoder over the window is also reported.
- DEC is supervised on pa, so its T2 is a ceiling for a linear readout of rates, not a competitor of the same kind. It enters T2 only.

## 5. Decision rules (from README, restated)
- "Tie" (declared in the paper-facing text): the 95% bootstrap CI of the paired difference lies inside +/-0.10 in d1 (T1) or +/-15 ms (T2); "FT better": CI excludes 0 and, in the family, BH-adjusted p < 0.05; otherwise "unresolved".
- FT ~ FT0 ~ POLb: the dip is polarization (expected). FT > FT0 and POLb: coupling-corrected fields carry extra timing information. TRAJ ~ FT: standard population methods see it too. FT > FR but not dFR/DEC: bidirectional rate changes cancel in the average. No uniqueness claim is sought; ties are reported as ties.

## 6. Method validation (separate, labelled as such)
Synthetic spike data with a planted field dip, and with J = 0, confirm that the extremum search and tests recover the dip and that FT vs FT0 shows no difference. This is method validation and is not evidence about the brain.

## Amendments
- **A1 (2026-10-12, before any real-data computation).** Smoothing is applied once per signal: FT, FT0 and FR are smoothed at the output (they are linear in the reach-averaged spins). The nonlinear PSTH indices (dFR/POLa, POLb, TRAJ) are computed from the PSTH smoothed with the same Gaussian (sigma = 3 bins) and are not smoothed again. The decoder's standardisation (per-cell mean/SD) uses training reaches only. For N < 4 conditions TRAJ uses k = min(3, N).
- **A2 (2026-10-12, after the synthetic method validation, still before any real-data computation).** The validation on synthetic data with NO planted dip (J = 0) gave a T3 depth ratio of about 1.7x for FT and 1.6x for FT0 (and ~1.0 for the model-free indices), because the pre-registered basis has 50-ms knots near the reach and 250-ms knots elsewhere, so the fitted field has more freedom (more fitted noise) in the real window than in the rest-control windows. Two sensitivity signals are added: FTu and FT0u, the field terms from M1/M0 fitted with a uniform basis (knots every 5 bins over the whole trace). They are not in the primary family. They answer whether the depth/timing advantage of FT survives equal flexibility in real and control windows. The primary analysis (non-uniform basis) is unchanged and is reported as pre-registered, together with this null calibration.
