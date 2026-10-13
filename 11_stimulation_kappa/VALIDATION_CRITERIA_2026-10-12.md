# E11 synthetic-validation pass criteria (written 2026-10-12, before any validation run)

SYNTHETIC DATA, method validation only. No real data is touched.

## Setup (fixed now)
- N = 12 neurons, 3 blocks, R = 60 reaches per block, T = 700 bins, exact ML (`code/stimfit_model.py`).
- Truth: J_ij ~ N(0.03, 0.12) symmetric; baseline fields h_i(t) = h0_i − c_i·exp(−((t−400)/40)²), h0_i ~ N(0.3, 0.15),
  c_i ~ U(0.2, 0.8), same in all blocks (a reach-locked rate dip).
- Laser onset per reach in stim_1 is 400 or 401 (random). Pulse window = onset … onset+5 (6 bins = 60 ms).
  Control window = bins 401 … 406 in stim_0 (median latency of 1 bin). κ is fitted for both windows; the fitted
  model has block-specific B-spline fields (`bspline_basis(400)` knots), per-window intercepts, shared J.
- 8 replicates per scenario (seeds 0–7). Defaults: lamB 1e-5, lamJ 1e-4, lamD 1e-3, lamK 1e-3.

## Scenarios
| ID | Truth in the stim_1 pulse window | Purpose |
|---|---|---|
| A | κ = +0.5, rates unchanged | recover a true coupling change |
| B | κ = 0, fields +0.5 on every neuron, ramping on over onset…onset+30 with a raised cosine | rate-only stimulation: κ must stay ≈ 0 |
| C | κ = 0, fields +0.8 on every neuron inside the 6-bin pulse window only | rate change confined to the window: no false positive |
| D | nothing changes | null |
| E | κ = +0.5 together with the rate change of B | recover κ despite a rate change |

## Statistic
Δκ = κ̂_pulse − κ̂_control (the primary S1 quantity), per replicate.

## Pass criteria (all must hold; failing any one is reported as a failure, not tuned away)
1. A and E: mean Δκ over the 8 replicates lies in [0.40, 0.60].
2. B, C, D: |mean Δκ| ≤ 0.10, and at least 7 of 8 replicates have |Δκ| < 0.30 (0.30 is the planned equivalence bound).
3. All scenarios: |mean κ̂_control| ≤ 0.10.
4. At least 95% of fits report convergence.
5. Median over fits of corr(Ĵ, J_true) over pairs ≥ 0.90.

Any change to this document after the first validation run is made as a dated amendment below, never by editing above.

## Amendment 1 (2026-10-12, before the validation run; no criterion changed)
A development smoke fit (scenario A, seed 0, `lamK = 1e-3`, 300 iterations) returned Δκ = 0.21 and did not converge.
Refits of the same data: 1500-iteration cap, `lamK = 1e-3` → Δκ = 0.21 (converged: the ridge, not the iteration cap, is the cause);
`lamK = 0` → Δκ = 0.44; `lamD = 1e-5` made no difference. The ridge on κ shrinks it because κ is one scalar with plenty of
data behind it. Defaults changed to **`lamK = 0`** and **maxiter = 800**. The criteria and scenarios above are unchanged,
and all five scenarios are run with the new defaults. Those four smoke fits are development only and are not counted.

## Result of run 1 (2026-10-12, defaults lamD = 1e-3, lamK = 0; cluster job 4362873; `assets/tables/validation_run1_*`)
Passed: criteria 1 (A 0.447, E 0.472), 3, 4 (100% converged), 5 (median J corr 0.9996); B (mean Δκ 0.074, 8/8 under 0.3) and D (0.027, 8/8).
**Failed criterion 2 in scenario C** (rate change confined to the pulse window): mean Δκ = 0.43, sd 0.29, only 1/8 replicates under 0.3.
This is a false positive of the model as specified. Overall: FAIL.

## Amendment 2 (2026-10-12, after run 1; criteria and scenarios unchanged)
Diagnosis: same mechanism as amendment 1. The window intercepts d_g exist to absorb uniform rate shifts, but their ridge
(`lamD = 1e-3`, in per-bin-normalised units) is large against the information in a 6-bin window, so d_g is shrunk and the
unabsorbed rate shift leaks into κ. Change: **`lamD = 1e-5`** (equal to `lamB`). Nothing else changes.
All five scenarios are rerun with the same seeds (run 2). Run 1 stays on record; both runs are reported. If run 2 fails
any criterion, the failure is reported as is.

## Result of run 2 (2026-10-12, defaults lamD = 1e-5, lamK = 0; cluster job 4362884; `assets/tables/validation_run2_*`)
**All five criteria pass.** Mean Δκ: A 0.451 (sd 0.066), E 0.476 (0.060); B 0.080 (0.101), C 0.025 (0.127), D 0.021 (0.041);
mean κ̂_control within ±0.03 everywhere; 8/8 replicates under |Δκ| < 0.3 in B, C, D; 40/40 fits converged; median corr(Ĵ, J) = 0.9996.
Caveats to carry forward: recovered κ is about 10% below truth (0.45 vs 0.5), and rate-only scenario B shows a small positive bias (+0.08).
Both are inside the criteria and well inside the planned 0.3 equivalence bound, but they are a reason not to claim precision below about 0.1.

## Amendment 3 (2026-10-12, after run 2, BEFORE run 3; Orchestrator-approved design changes + a finding about real reach counts)
What changed in the design, and why the validation is rerun once (run 3) with the same seeds:
1. **Control window.** Not fixed at bins 401–406: each control-block (stim_0) reach gets a pseudo-onset drawn from the session's
   own stim_1 onset distribution, window = pseudo_onset … +5 (real data: floor of onset_float, seed recorded; validation:
   drawn from the replicate's two onset values, 400/401, with an independent generator so the simulated data are identical to run 2).
2. **Fitted bin range 300–559** (instead of 0–699). Reason: cost. An N = 16 evaluation is 1.3 s with 780 units vs 3.4 s with 2100; the pulse window
   (about bins 375–424) sits well inside, and block-specific B-spline fields are unchanged inside the range.
3. **Real reach counts are much lower than the 60 used so far.** Non-excluded reaches per block in the 21 sessions are about 9–29
   (stim_1: 9–25). Run 3 therefore adds a **session-sized set: R = 18 reaches per block, 16 replicates per scenario**, next to the original R = 60 set (8 replicates).
4. Likelihood code now does one `exp` per entry (same values; unit tests re-run: 2.3e-13 vs brute force, gradient error 3e-8).

Pass criteria for run 3:
- **R = 60:** criteria 1–5 exactly as above (the design changes 1–2 must not break them).
- **R = 18** (power set; looser because the estimator is noisier): the same statistic Δκ per replicate; (1) A, E: mean Δκ in [0.35, 0.65];
  (2) B, C, D: |mean Δκ| ≤ 0.12; (3) |mean κ̂_control| ≤ 0.12; (4) convergence ≥ 95%; the per-replicate rule is dropped. The sd of Δκ is **reported, not tested**:
  it is the single-session precision and drives the power statement for n = 21 sessions.
Failing any criterion is reported as a failure.

## Result of run 3 (job 4362993; `assets/tables/validation_run3_*`): FAIL
Parameter recovery was fine where the fit finished (R = 60: A 0.444, E 0.483, B 0.077, C −0.024, D 0.023; R = 18: A 0.465, E 0.483, B −0.009, D −0.005, sd of Δκ 0.12–0.15),
but **criterion 4 failed badly: 117 of 120 fits hit the 800-iteration cap without converging** (runs 1–2: 100% converged in about 400). R = 18 scenario C
also failed criterion 2 (mean Δκ 1.33, sd 5.4) because of a single runaway unconverged fit (κ_pulse = 21).

## Amendment 4 (2026-10-12, after run 3, BEFORE run 4; criteria unchanged)
Diagnosis: run 3 cropped the E09 basis (`bspline_basis(400)[300:560]`), which leaves edge spline columns that are only partly inside the range;
they are weakly determined and slow the optimiser (and let κ run away when the window information is small). Change: **a basis built directly on bins 300–559**
(`bspline_basis_crop`: 25-bin knots 300–340, 5-bin knots 340–559, clamped ends). Development smoke on the three worst cases (C/R18/seed 7, A/R60/seed 0, D/R18/seed 0):
all converged in 322–362 iterations (about 30 s). Run 4 repeats the whole run 3 design with the same seeds; criteria 1–5 and the R = 18 criteria as in amendment 3.

## Result of run 4 (job 4363017; `assets/tables/validation_run4_*`): PASS
R = 60 (8 replicates): all of criteria 1–5 pass. Δκ: A 0.445, E 0.485, B 0.076, C −0.015, D 0.024; 120/120 fits converged; median corr(Ĵ, J) 0.999.
R = 18 (16 replicates, session-sized): all criteria pass. Δκ: A 0.465 (sd 0.12), E 0.482 (0.15), B −0.004 (0.13), C −0.002 (0.27), D −0.004 (0.14); median corr(Ĵ, J) 0.995.
Single-session precision at session-sized R: sd of Δκ ≈ 0.12–0.15 (0.27 when a rate change is confined to the window). Over n = 21 sessions that is a standard error of about 0.03–0.06.

## Amendment 5 (2026-10-12, after run 4, BEFORE run 5; criteria unchanged): optimiser tolerance 1e-7
Real-data smoke fits (session 210511, N = 12, see PREREGISTRATION amendment A2) needed 640–800 iterations (synthetic: about 330) and one shuffle-null refit hit the
800-iteration cap with a runaway κ (1.6). Diagnosis on that session (primary design + that shuffle, 4 settings): optimiser tolerance `ftol` 1e-9 → 1e-7 cuts iterations 798 → 339,
changes the log-likelihood by 7 nats of 278,700 and κ by ≤ 0.02, and the shuffle refit then converges. A stronger spline ridge (`lamB` 1e-4) was rejected: it changes the model
(log-likelihood −2,400 nats, κ in both windows +0.3). **Change: default `tol = 1e-7`.** Run 5 repeats run 4 (same seeds, same criteria) with this tolerance.

## Result of run 5 (job 4363048, tol 1e-7; `assets/tables/validation_run5_*`): PASS
Identical conclusions to run 4. R = 60: Δκ A 0.448, E 0.481, B 0.075, C −0.009, D 0.024; R = 18: A 0.463 (sd 0.12), E 0.484 (0.15), B −0.005 (0.13), C −0.007 (0.28), D −0.003 (0.14);
120/120 converged (mean 171 iterations, 38 s per fit); median corr(Ĵ, J) 0.999 / 0.995. This is the final fit configuration.

## Amendment 6 (2026-10-12, BEFORE run 6; from the independent code review, `REVIEW_2026-10-12.md`, and PREREGISTRATION amendment A3)
Runs 1–5 validated a configuration that is easier than the real analysis: onsets only 400/401, identical fields in every reach and block, no shuffle null, and (found afterwards) a different fitted range from the real run.
Run 6 validates the **actual real-data pipeline** (`run_stimfit.make_design / build_groups / shuffle_blocks / crop`, range 300–559, tol 1e-7) on synthetic sessions built to look like real ones:
- **Structures** (3 real sessions, only their shape is used): 210421, 210511, 210514 → number of reaches per block (non-excluded) as in the real session, the **real stim_1 onset bins** (jitter 375–418), N = min(N, 12) neurons (fixed random subset). Block-specific fields h_b(t) = arctanh(clip(2p − 1)) from the real **smoothed PSTH** of each neuron in each block (Gaussian σ = 3 bins) — they contain the real reach-locked rate changes and the real stimulation rate response. J_ij ~ N(0.03, 0.10) symmetric.
- **Per-reach common gain** g_r ~ N(0, 0.25) added to all fields of reach r (all blocks): a shared trial-to-trial rate fluctuation that creates cross-neuron covariance without coupling. This is the rate confound that the null correction is meant to remove.
- **Scenarios** (all with the real PSTH fields and the gain): A κ_pulse = +0.5; B κ = 0 with an extra field ramp +0.5 (raised cosine over 30 bins from each reach's onset); C κ = 0 with +0.8 in the 6-bin window only; D κ = 0 (null, only the real rate structure); E κ = +0.5 with the ramp of B. κ_control = 0 in all.
- **Fit:** exactly the real pipeline (primary design, pseudo-onset control from the replicate's own stim_1 onsets), plus **10 aligned shuffle refits** per dataset → null means → null-corrected Δκ_c = (κ_p − null κ_p) − (κ_c − null κ_c); raw Δκ_r = κ_p − κ_c. 3 structures × 5 scenarios × 8 replicates = 120 datasets.
- **Pass criteria** (pooled over structures, 24 datasets per scenario; all must hold):
  1. A and E: mean Δκ_c in [0.35, 0.65].
  2. B, C, D: |mean Δκ_c| ≤ 0.10.
  3. D: |mean raw κ_control| ≤ 0.10 (the real run had |κ_control| up to 1 under the wrong range; this checks the corrected range does not).
  4. At least 95% of fits converge.
  5. Null-correction check, reported and tested: in B, C and D pooled, |mean Δκ_c| ≤ |mean Δκ_r| + 0.02 (the aligned null must not make the bias worse).
  - **Reported, not tested:** sd of Δκ_c and Δκ_r per scenario and structure; sd of the 10 null κ values per window (to compare with the real refit's shuffle spread, the Orchestrator's noise-match request); fraction of fits with κ < −1.
- Failing any criterion is reported as a failure.

## Amendment 6b (2026-10-12, written BEFORE run 6 has reported, in response to the Orchestrator's question about the aligned null)
The aligned shuffle shifts each donor reach by (receiving onset − donor onset) bins. This aligns the laser-locked response and misaligns the reach-locked one by the same shift.
Shift sizes on the real data (`assets/tables/shuffle_shift_sizes.csv`, all donor–receiver pairs of non-excluded stim_1 reaches): mean 1.8 bins, median 1, 95th percentile 5, maximum 20;
81% of pairs shift by ≤ 2 bins, 96% by ≤ 5. The wrap of the circular shift touches only the first and last 20 bins of the 700-bin array (the fit range starts at bin 300), so no wrapped data enter the fit;
the shift does bring contiguous data from up to 20 bins outside 300–559 into the range (by design).
The three structures of run 6 (210421, 210511, 210514) have mild onset jitter (ranges 398–406, 399–402, 400–406; mean shift 0.7–1.9 bins). The trade-off only matters in high-jitter sessions, so:
- **Run 6b** repeats run 6 (same code, same scenarios, 8 replicates) on three **high-jitter structures**: 220519 (onsets 374–394, 25 reaches), 210608 (399–417, 10 reaches), 220520 (378–396, 9 reaches).
- **Criteria 1–5 of amendment 6 apply separately to the low-jitter group (run 6) and the high-jitter group (run 6b).** The null scenario D in the high-jitter group is the direct test of whether the reach-locked misalignment hurts (|mean raw κ_control| ≤ 0.10; |mean Δκ_c| ≤ 0.10; correction not adding bias).
- If the high-jitter group fails and the low-jitter group passes, the failure is reported and the fallback (shuffle only within reaches whose onsets lie within ±2 bins) is **proposed to the Orchestrator, not applied** until agreed.

## Result of run 6 (job 4363385, low-jitter group: 210421, 210511, 210514; `assets/tables/validation_run6_*`): FAIL (criterion 1)
Pooled over 24 datasets per scenario (null-corrected Δκ_c / raw Δκ_r; sd in brackets): A 0.154 (0.34) / 0.383 (0.19); B 0.036 / 0.019; C 0.036 / 0.168; D −0.082 / 0.000 (0.30 / 0.18); E 0.295 (0.35) / 0.333 (0.21).
Criterion 1 **fails** (A and E recover 0.15 and 0.30 after null correction, required 0.35–0.65). Criteria 2, 3 (|κ_control| in D = 0.009) and 4 (100% converged) pass; criterion 5 passes (pooled |bias| 0.051 corrected vs 0.062 raw). Median corr(Ĵ, J) 0.98.
Diagnosis: the aligned shuffle null is **not centred at zero when the true κ is non-zero**: mean(null κ_pulse − null κ_control) is +0.228 in A (3.8 standard errors from zero), +0.132 in C, +0.083 in D, +0.038 in E, −0.017 in B.
So the correction subtracts part of the true effect and adds variance (sd 0.34 vs 0.19). The rate-confound bias that motivated it (+0.08 in the first validation's scenario B) is not present in this more realistic simulation: raw Δκ is 0.019 in B and 0.000 in D; the one raw bias is +0.17 in C (a rate change confined to the 6-bin window).
Also reported: the shuffled fits have heavy tails (fits with κ < −1: A 8, B 19, C 11, D 10, E 16 of 264 per scenario), synthetic null sd of κ 0.2–0.5 per window.
No change to the analysis has been made in response to this (the Orchestrator asked to be told first). Options are proposed in the message to the Orchestrator. Run 6b (high-jitter structures, amendment 6b) is run unchanged.

## Result of run 6b (job 4363399, high-jitter group: 220519, 210608, 220520; `assets/tables/validation_run6b_*`): FAIL (criteria 1, 2, 5)
Pooled over 24 datasets per scenario (null-corrected Δκ_c / raw Δκ_r; sd in brackets): A 0.350 (0.34) / 0.438 (0.26); B 0.038 / −0.021; C 0.297 / 0.250; D 0.105 (0.32) / 0.009 (0.26); E 0.319 (0.41) / 0.282 (0.22).
Criterion 1: E fails (0.319 < 0.35), A passes narrowly (0.350). Criterion 2: **C fails (0.297) and D fails (0.105 > 0.10)**. Criterion 3 passes (|κ_control| in D = 0.036). Criterion 4 passes (100% converged). Criterion 5 **fails**: pooled |bias| in B, C, D is 0.147 corrected vs 0.093 raw (+0.02 allowance), i.e. **the null correction adds bias in the high-jitter group**.
In the null scenario D the null difference (null κ_pulse − null κ_control) is negative (−0.105, −0.122, −0.062 in the three structures) while raw Δκ is ≈ 0 (0.129, 0.016, −0.119): the null subtracts a spurious negative number. This is the pattern expected if the reach-locked misalignment of the shifted shuffle matters when onset jitter is large.
Raw Δκ in this group: A 0.438, B −0.021, D 0.009 are fine; **C (rate change confined to the 6-bin window) leaks +0.25** and E (κ plus a rate ramp) recovers 0.28.
Combined with run 6: the null-corrected statistic fails the pre-registered criteria in both groups; raw Δκ recovers κ and has no bias in B and D in both groups, with a known positive bias in C (+0.17 low-jitter, +0.25 high-jitter). No analysis change has been made; the decision is with the Orchestrator and Gunnar.

## Decision after runs 6 and 6b (2026-10-13; PREREGISTRATION amendment A4)
The primary statistic becomes the raw Δκ; the aligned shuffle null remains as a labelled diagnostic (10 shuffles per subset). **Validation 7** (shuffle only within onsets ±2 bins) is **not run**: it would address the reach-locked misalignment of run 6b but not the off-centre null of run 6 (mild jitter), so it cannot rescue the null as a corrected statistic. Possible future check of the diagnostic only.
The pre-registered criteria for the raw statistic are not separately stated; its observed behaviour in runs 6 and 6b is reported in full (A 0.38–0.44, B ≈ 0, D ≈ 0, C +0.17 to +0.25, E 0.28–0.33).
