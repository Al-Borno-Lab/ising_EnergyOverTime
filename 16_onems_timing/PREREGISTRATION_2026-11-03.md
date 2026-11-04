# E16 pre-registration (2026-11-03, frozen before any E16 code touched the data)

Approved by Gunnar on 2026-11-03, after his PI proposed the 1-ms energy figure as a headline result.

## Claim under test
In the 1-ms fit, the field term of the population energy shows:
- a **peak at peak acceleration**;
- a **trough at peak velocity**.

The claim comes from the session-median curve of the exploratory 1-ms refit (evidence package S3, panel D).

## What has already been seen (disclosure)
- **Exploratory 1-ms fits** (standard basis: 50-ms knots from −600 to +1600 ms around peak velocity, 250-ms knots elsewhere; same 63 conditions). The median curve shows a peak near the median peak acceleration (−40 ms) and a trough near peak velocity.
- **Two per-session checks on those fits, run on 2026-11-03 before this document. Both were negative:**
  1. **Extremum timing.** Search window −200 … +100 ms, 5-ms smoothing.
     - The peak fell within ±30 ms of the condition's own peak acceleration in 2/63 conditions; a rest window gave 6/63.
     - The trough fell within ±30 ms of peak velocity in 14/63; rest gave 15/63.
     - The peak preceded the trough in 11/21 sessions.
  2. **Shape.** Rise into peak acceleration (mean −50…−20 ms minus −120…−80 ms) in 7/21 sessions; fall into peak velocity (−10…+20 minus −50…−20) in 10/21; full rise-then-fall in 0/21 (rest: 5/21).
- **Concern.** A peak-to-trough swing within about 40 ms is at the scale of the 50-ms knots, so it could be spline ringing. Unequal knot density between reach and rest windows also inflates reach-window features (E12, R7).

This test therefore changes the basis. Its primary analysis uses bases with **equal flexibility everywhere**, plus a knot-offset control. The prior from the checks above is negative. A positive result here would mean the standard-basis checks were too noisy, not that the effect was found by searching.

## Data
- **Spikes:** 1-ms binary spikes from the E14 extraction (`14_binsize_1ms_synchrony/results/spk/<session>.npz`, `c1`). The extraction passed gates G1–G3 in 21/21 sessions; 1-ms re-bins equal the 10-ms data exactly.
- **Reaches and blocks:** all three blocks, with `exclude == 1` reaches dropped, giving 63 conditions.
- **Landmarks:** `09_claude_science_handoff/assets/tables/phase_landmarks.csv`. Peak velocity pv = 10·`zero_cross` ms and peak acceleration pa = 10·`peak_acc` ms, both within the 7000-ms window; 10-ms resolution.

## Model and bases
M1 (time-varying fields, fixed symmetric J), fitted by pseudo-likelihood as in `tvfield_model` (lamB = 1e-5, lamJ = 1e-3), on all reaches of a condition. Three bases for h_i(t), all cubic B-splines:

| Basis | Knots | Role |
|---|---|---|
| U0 | every 50 ms over the whole 7000 ms, starting at 0 | **primary** |
| U25 | every 50 ms, offset by 25 ms | ringing control |
| STD | 50 ms from pv −600 to +1600 ms, 250 ms elsewhere (the S3 basis) | secondary, reproduces the exploratory figure |

**Field term per condition:** FT(t) = −(1/N) · mean over reaches of Σ_i h_i(t) s_i(r, t), with s = ±1 and 1-ms bins. Gaussian smoothing, σ = 5 ms.

## Tests (per condition, then the session mean over its 3 conditions)
- **Search window:** pv −200 … +100 ms.
  - t_peak = argmax of FT in the window; t_trough = argmin.
  - **Peak hit:** |t_peak − pa| ≤ 30 ms. **Trough hit:** |t_trough − pv| ≤ 30 ms.
- **Rest control:** the same search, with the window and both landmarks shifted by −2000 ms and by −2500 ms. The rest hit rate is the mean over the two shifts.
- **Statistics:**
  - D_peak and D_trough = session hit fraction (reach) minus session hit fraction (rest).
  - Two-sided Wilcoxon signed-rank over 21 sessions.
  - Hodges–Lehmann estimate with a bootstrap 95% CI (10,000 resamples, seed 16).
- **Primary family** (BH-FDR, q = 0.05): D_peak and D_trough under U0.
- **Decision rule:** the claim is **supported** if both primary tests are positive and q < 0.05, **and** both are positive with p < 0.05 (unadjusted) under U25. It is **partly supported** if only one landmark meets this. Otherwise it is **not supported**.
- **Secondary:**
  - the same tests under STD;
  - median timing errors |t_peak − pa| and |t_trough − pv|, reach vs rest;
  - the fraction of sessions in which the peak precedes the trough;
  - cohort (2021 vs 2022);
  - stim 0 only.

## Synthetic validation (must pass before real results are read)
**Ground truth:** for 6 conditions (210421_0, 210511_1, 210515_2, 220516_0, 220519_1, 210614_0, with N from 12 to 25), take the U0 fit of the real data. Its fields are smoothed with a 100-ms Gaussian to remove sharp features. J is the real fitted J. Data are simulated at 1 ms by Gibbs sampling (200 sweeps; real R).

| Scenario | Fields | Pass criterion |
|---|---|---|
| Z0 (null) | smooth fields | under U0, the reach − rest hit-fraction difference for peaks and for troughs, averaged over the 6 conditions, lies within ±0.2 |
| Z1 (planted) | each cell's field gets δ_i(t) = 0.3 · σ_i · [g(t − pv) − g(t − pa)], where g is a Gaussian with SD 15 ms and σ_i = sign of the cell's mean spin in the search window under the truth model. Because FT = −(1/N)Σ h s, this raises FT at pa and lowers it at pv. This puts an FT peak at pa and a trough at pv | under U0, peak hits ≥ 4/6 and trough hits ≥ 4/6 |

Z0 is also analysed under STD and reported. This shows whether the standard basis creates a peak/trough on its own.

## Review
The analysis code and `stats_e16.py` are reviewed by an independent agent before real results are read.

## Amendment A1 (2026-11-03, after independent code review, before any real-data result was read)
The review found no blockers. Changes:
1. **"Positive" has one definition everywhere:** Hodges–Lehmann > 0 **and** W+ > W− (Wilcoxon signed-rank sums of the nonzero session differences), for both U0 and U25.
2. **BH-FDR is computed over the finite primary p-values,** with the family size kept at 2. Before, one undefined p-value (all differences zero) would have blanked both q-values.
3. **Timing-error rows** now also report the reach and rest median errors.
4. **Booleans** are written as true/false in the summary.

Noted, no change:
- the multiprocessing start method relies on fork, which is the default in the container's Python 3.10;
- the pv landmark is the start of its 10-ms bin (a fixed −5 ms offset for trough errors), as pre-registered.

## Amendment A2 (2026-11-03): decision after the synthetic gate; real results unread at the time of writing
**Synthetic result.** Z1 passed: 5/6 peak hits and 5/6 trough hits, median errors 8.5 and 9.5 ms. Z0 failed for peaks: D_peak = −0.33, outside ±0.2; for troughs D_trough = +0.08, within. Cause: with no planted peak, the broad reach-locked dip pushes the reach-window maximum to the window edge. The rest comparison is therefore biased **against** peak hits; it is conservative for the peak test. The trough test passes.

**Decision (Gunnar, 2026-11-03).** E16 is read internally with this bias stated, and **is not reported in the paper**:
- a positive D_peak would be conservative evidence;
- a null or negative D_peak is uninterpretable;
- D_trough is interpretable as registered.
