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
