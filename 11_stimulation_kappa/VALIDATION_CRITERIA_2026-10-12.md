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
