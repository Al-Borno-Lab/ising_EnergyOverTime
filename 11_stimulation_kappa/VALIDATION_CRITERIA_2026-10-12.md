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
