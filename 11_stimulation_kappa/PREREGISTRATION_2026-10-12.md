# E11 pre-registration (dated 2026-10-12, frozen before any real-data model fit)

**What has touched real data so far:** only the extraction of laser times (`ReachS.stimtime`) and the tabulation of neuron and reach counts per session. **No κ, J or field has been fitted on real data.**
Later changes are dated amendments at the end; none may be made after real-data results exist except to document a deviation.
Synthetic validation (method only): `VALIDATION_CRITERIA_2026-10-12.md`, runs 1–3 (summary in section 9 below).

## 1. Question and estimand
Does closed-loop optogenetic stimulation change how Purkinje cells interact (coupling), or only how much they fire?
Estimand: a scalar κ in the pulse window, J_eff = (1 + κ)·J, with J shared by all three blocks of a session and rate changes absorbed by
block-specific fields and window intercepts.

## 2. Data
- `spikes_v2.npz` (21 sessions × 3 blocks, 10-ms bins, binarised, 700 bins), reaches with `exclude_k == 0` (primary). N = 3–32 cells; 9–29 reaches per block (stim_1: 9–25).
- Laser time per stim_1 reach: `ReachS.stimtime` (absolute time, s) → `onset_float = (stimtime − t_k[r,0]) / 0.01`; table `assets/tables/laser_onsets.csv`.
  All 354 non-excluded stim_1 reaches have exactly one laser time. Onset median 400.0, IQR 398–401, range 375–418.
- Cohorts: 2021 = 15 sessions (fChR2, fChR4, fChR5, cChR1), 2022 = 6 sessions (DJC000, DJC002). Six mice.
- Open lab question (not resolved): whether all mice are opsin-positive. If any session is opsin-negative or red-light, it is a negative control and is analysed as such, not pooled.

## 3. Model (code: `code/stimfit_model.py`, tested against brute force)
P(s) ∝ exp(h_b(t)·s + ½ a_{r,t} sᵀJs), s ∈ {−1,+1}^N; a = 1 + κ_g inside window group g, else 1; h_b(t) = B(t)W_b (+ d_g inside group g).
- Basis: cubic B-splines built on bins 300–559 (`bspline_basis_crop` in `code/tvfield_model_basis.py`): 25-bin knots 300–340, 5-bin knots 340–559, clamped ends (the 5-bin spacing is that of E09 `bspline_basis(400)`). 260 bins per block.
- Penalties (per-bin-normalised): λ_B = 1e-5 on W, λ_J = 1e-4 on J, λ_D = 1e-5 on d_g, **λ_κ = 0**. L-BFGS-B, ≤ 800 iterations, exact enumeration.
- Convergence rule: a non-converged fit is rerun once from its last iterate with 2000 iterations (`reran` is stored); if it still has not converged it is excluded and listed. A session needs ≥ 5 converged subsets (N > 16) or a converged fit (N ≤ 16), else it is reported as failed and left out of S1–S3.

## 4. Windows
- **Pulse** (stim_1, per reach): bins `floor(onset_float)` … +5 (60 ms). Primary rounding: floor (the bin containing the laser time).
- **Control** (stim_0, per reach): a pseudo-onset drawn with replacement from that session's stim_1 `floor(onset_float)` values, window pseudo-onset … +5.
  Seed `[20261012, session_date, 7]`, the same draw for all fits of a session. This puts control windows at the same reach phase as pulse windows.
- Windows are clipped at bin 699 (never reached in practice).

## 5. Neurons, subsets and nulls
- N ≤ 16: all neurons, exact. N > 16 (12 sessions): **10 random 16-neuron subsets**, seed `[20261012, session_date]`, subsets stored with every result.
- **Null:** 20 trial-shuffle refits per (session, subset): for each block, each neuron's reach order is permuted independently (seed `[20261012, session_date, subset, 100+k]`). The same subsets are used for real fits and nulls.
- Per (session, subset): Dκ_raw = κ_pulse − κ_control; **null-corrected Dκ = (κ_pulse − mean_null κ_pulse) − (κ_control − mean_null κ_control)**.
  Session value = median over subsets (the value itself when N ≤ 16).
- Reason for the correction: synthetic validation shows a small positive bias of +0.08 under rate-only stimulation. **Effects smaller than about 0.1 in |κ| are not interpretable** and will not be described as findings.
- Exactness check (Orchestrator requirement): in session **210515 (N = 18)** one full 18-neuron exact fit (no shuffles) is compared with the median over its 10 subset fits; reported as a robustness note, with no decision rule attached.

## 6. Hypotheses and tests (unit = session, n = 21)
**S1 (primary).** Null-corrected Dκ per session; two-sided Wilcoxon signed-rank against 0.
- Effect size: Hodges–Lehmann median with a 90% CI (distribution-free, from the Walsh averages).
- **Equivalence:** "coupling is invariant to stimulation" is concluded only if the 90% CI lies entirely inside (−0.3, 0.3) (two one-sided tests at 0.05). p > 0.05 alone never supports invariance.
- Outcomes: (a) Wilcoxon p_FDR < 0.05 and |HL| ≥ 0.1 → "stimulation changes coupling"; (b) equivalence met → "invariant within ±0.3"; (c) otherwise → inconclusive, and the CI is reported.

**S2.** Fixed-J (κ = 0) thermodynamics at the fitted fields, exact: ΔI/N, the multi-information per neuron in bits (primary S2 metric), pulse window (stim_1, with d_pulse) minus matched control window (stim_0, with d_control).
Session value = median over subsets; two-sided Wilcoxon. Δχ_J/N is computed and reported but is not part of the test family. Prediction (PAPER_ANALYSIS_PLAN step 10): ΔI/N and Δχ_J/N fall in the pulse window while κ ≈ 0. Direction is reported whichever way it goes.

**Multiplicity:** Benjamini–Hochberg over {S1, S2} (2 tests).

**S3 (exploratory, not in the family, no FDR).** One fit per subset with groups: stim_1 first half of reaches (early pulse), second half (late pulse), stim_0 control, stim_2 washout (pseudo-onsets from the same distribution).
Tests: κ_early − κ_late, and κ_washout − κ_control, Wilcoxon over sessions. Raw differences (the shared bias cancels).

## 7. Sensitivity analyses (reported next to the primary; none replaces it)
1. Onset rounding: `round(onset_float)`, and `floor + 1`.
2. Fixed control window, bins 401–406 in stim_0 (instead of pseudo-onsets).
3. All reaches (exclude flag ignored), using the laser times of excluded reaches where they exist.
4. Drop sessions with fewer than 5 cells (210622: 3 cells, 210623: 4 cells → n = 19).
5. Cohort split: 2021 (n = 15) and 2022 (n = 6) separately, same tests.
6. Mouse as a random effect: linear mixed model of session-level null-corrected Dκ ~ 1 + (1 | mouse), REML, `statsmodels.MixedLM`; report the intercept and its CI.
7. Raw (uncorrected) Dκ.
Sensitivity fits are not null-corrected beyond what is stated; they are compared by the HL estimate and CI.

## 8. Known limits (stated now)
- Reaches per block are few (9–29), so single-session κ is imprecise (see section 9); the test is across sessions.
- Cell stability across blocks was not checked; PC identity follows the export (putative PCs included).
- κ is a single scale on all pairs in the window; pair-specific changes are not tested.
- Binarisation (count > 0) is the pipeline's convention.

## 9. Synthetic validation summary (method validation only)
Full record: `VALIDATION_CRITERIA_2026-10-12.md` (four runs, four dated amendments, criteria written before the runs). Runs 1 and 3 FAILED and are kept (run 1: ridge on window intercepts too strong, false positive in scenario C; run 3: cropped-basis edge columns, no convergence). Run 4 is the design used here and passes everything.
Scenarios (all SYNTHETIC): A κ = +0.5; B rate-only ramp; C rate change confined to the 60-ms window; D null; E κ = +0.5 plus the rate ramp. N = 12, windows and fitting range as in sections 3–4.
- Session-sized data (R = 18 reaches per block, 16 replicates): Δκ = 0.465 (A), 0.482 (E), −0.004 (B), −0.002 (C), −0.004 (D). Single-session sd of Δκ is 0.12–0.15 (0.27 in C). Standard error over 21 sessions about 0.03–0.06.
- Larger data (R = 60, 8 replicates): 0.445 (A), 0.485 (E), 0.076 (B), −0.015 (C), 0.024 (D); 120/120 converged.
- Consequences adopted here: recovered κ is about 7–10% below truth; rate-only data can bias Δκ upward by about 0.08 (hence the null correction and the "not interpretable below 0.1" rule); an equivalence bound of 0.3 is well resolved at n = 21, an effect near 0.1 is only marginally resolved.
- Not validated: N > 12 (real N is up to 16 per fit), real onset jitter (375–418; validation used 400/401), and the shuffle-null correction itself (applied only to real data).

## Amendments
(none yet)

### Amendment A1 (2026-10-12, before any real-data fit)
`assets/tables/laser_onsets.csv` regenerated to keep excluded reaches with their `exclude` flag (1249 rows; the 1092 `exclude == 0` rows are identical to the previous table). Needed only for sensitivity analysis 3 (all reaches). Extraction only; no model was fitted. `run_stimfit.py` gained task filters (`--subsets`, `--nshuf`) used for the smoke test; no analysis choice changed.

### Amendment A2 (2026-10-12, before the full real-data run)
- **Fit settings.** Optimiser tolerance `ftol = 1e-7` (validation amendment 5; validation run 5 passes). Chosen on iteration counts and log-likelihood, not on Δκ.
  **Disclosure:** smoke and diagnosis fits on session 210511 (N = 12, subset all neurons) and the timing fit on session 220519 (subset 0) were run and their κ values were visible before this amendment.
  In 210511 the primary Δκ_raw was about 0.00 (κ_pulse −0.32, κ_control −0.32). Those fits are logged in `runs/RUN_LOG.md` as smoke/diagnosis and are **not** used as results; the same sessions are refitted in the full run.
  A stronger spline ridge (λ_B 1e-4) was tested on 210511 and rejected because it changes the model (log-likelihood −2,400 nats), not because of its κ.
- **Rerun rule implemented** as in section 3 (one rerun, 2000 iterations).
- **Null mean** needs at least 10 converged shuffles of 20 for a (session, subset), else that subset is dropped from S1 and listed.
- **Cost (measured).** One 16-neuron fit: 11.4 CPU-min (session 220519, 306 iterations); N = 12: about 3 CPU-min. Total for the full plan (21 sessions, all variants on all subsets, 20 shuffles): about 590 CPU-hours, i.e. about 3.1 h wall time per job as 3 jobs × 64 CPUs.

### Amendment A3 (2026-10-12, after the first full run, BEFORE any refit): correction of implementation errors
**What was seen before this amendment.** The first full run (run 1; jobs 4363060–62; commit 25c1136) was analysed and its results were read: S1 null-corrected Δκ HL −0.236 (90% CI −0.442 to −0.028, Wilcoxon p = 0.070, FDR 0.14), S2 ΔI/N p = 0.97, the cohort split (2021 −0.41, 2022 +0.28), the fixed-control sensitivity (−0.08), and the regularization table. An independent code review (`REVIEW_2026-10-12.md`) then found errors; the amendment is a response to the review and to the code, not to the κ values.
**What is unchanged:** the question, the model, the windows (pulse = onset … +5; control = per-reach pseudo-onset from the session's stim_1 onsets), the primary statistic (null-corrected κ_pulse − κ_control, session median over subsets), S1/S2/S3 and their decision rules, the equivalence bound ±0.3, BH over {S1, S2}, the sensitivity set, the subsets and seeds, fit settings (λ_B 1e-5, λ_J 1e-4, λ_D 1e-5, λ_κ 0, tol 1e-7, 800 iterations then one 2000-iteration rerun).
**What changes, and why:**
1. **Fitted range (implementation error against section 3).** Run 1 fitted bins 0–699 (`T_LO, T_HI = 0, 700` in `run_stimfit.py`) with a basis whose knots leave a 144-bin gap (555–699). The pre-registered range is bins 300–559. The runner now sets `PREREG_RANGE = (300, 560)`, asserts that the fitted range equals it in every task, and stores `t_range` in every output row. All S1–S3 and sensitivity numbers are refitted; **run 1 results are superseded** (kept as `*_run1_0to700.*`).
2. **Laser onsets (precision error).** `stimtime` was written with 6 significant digits (10 ms resolution above 1000 s) and `floor` of a float quotient put 134 of 354 windows one bin early. Fix: re-extract `stimtime` with 17 digits; the onset bin is `floor((stimtime − t_k[0]) / 0.01 + 1e-6)`. The primary rule "bin containing the laser time" is unchanged; the evidence that `t_k` marks the bin start is E14 gate G1 (left-closed edges [t, t+10 ms) reproduce Bin10 exactly, 0 mismatches in 3.57 M bins, session 210421).
3. **Shuffle null (design error, not what section 5 intended).** Run 1 permuted reaches per neuron but left each pulse window at the row's own onset, so a neuron's onset-locked response was moved relative to its window and the null did not preserve the rate structure the correction is meant to remove. The corrected null: permute reaches per neuron and block as before, and in stim_1 shift each donor reach in time by (onset of receiving row − onset of donor reach) so the response stays in the window (circular shift; the wrap is far outside 300–559). Stim_0/stim_2 are permuted as before. A per-neuron window mask is not possible (κ multiplies the whole population state in a bin); the time shift is the equivalent.
4. **Sensitivity variants:** in `round` and `shift1`, the control pseudo-onsets are now drawn from that variant's onsets (so control and pulse windows keep the same phase distribution).
5. **κ < −1** flips the sign of J_eff and is not meaningful as a "coupling scale". It is not bounded in the fit; the number of fits and sessions with κ < −1 is reported for every analysis.
6. **Validation** is rerun under harder conditions (`VALIDATION_CRITERIA_2026-10-12.md`, amendment 6) before the refit.
7. **No other analysis choice is changed.** If the refit differs from run 1, both are reported.
