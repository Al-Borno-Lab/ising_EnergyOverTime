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
- Convergence rule: a non-converged fit is rerun once with 2000 iterations; if it still has not converged it is excluded and listed. A session needs ≥ 5 converged subsets (N > 16) or a converged fit (N ≤ 16), else it is reported as failed and left out of S1–S3.

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
