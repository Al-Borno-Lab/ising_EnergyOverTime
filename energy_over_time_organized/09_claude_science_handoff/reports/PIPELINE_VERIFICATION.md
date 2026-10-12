# Pipeline verification from Dylan's raw source files (2026-10-07)

Source: `math-alderaan:/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_<date>_<mouse>_NPresults_short.mat` (24 files).
Each stage was re-implemented independently in Python (`reextract.py`, `verify_fit.py`) and compared with the files the pipeline used.

## Stage-by-stage result
| Stage | What was checked | Result |
|---|---|---|
| 1. Source → `*_bin10_spike.mat` (organizeData) | Unit selection (cellID == PC), reach selection by ReachS.stim, time masks, spike counts, kinematic interpolation and per-reach rescaling | **Exact match.** 21/21 sessions and 1249/1249 reaches; PC counts equal; bin times, spike counts and kinematics identical (max diff 1.6e-15) |
| 2. Binarization and window | Spikes in per_reach_state.csv vs bin10 counts > 0, rows 100:800, reach order | **Exact match** in 63/63 conditions |
| 3. Energy terms | Σhᵢsᵢ, ΣJᵢⱼsᵢsⱼ (i<j) and E = −(Σhs + ΣJss), recomputed from h_parameters.csv / J_parameters.csv | **Exact** (max diff 5e-14); J column order is upper-triangular as assumed |
| 4. Ising fit (MCH) | Do the stored (h, J) reproduce the data's mean activities and pairwise correlations? Exact enumeration for N ≤ 20 (39 conditions), Gibbs sampling above that | **Pass.** Median error in pair correlations 0.0013 rms vs a data covariance of 0.06 rms: 99.98% of the covariance reproduced (minimum 92%, in an N = 3 condition). Mean-activity error ≤ 0.023 |
| 5. Independent refit | Static pseudo-likelihood (PL) refit vs stored J | r = 0.99 median (0.996 in exact-check conditions). Lower agreement only in 210511 (a neuron firing in 99% of bins; the PL fit reproduces moments better there, as found earlier) |

**Conclusion: the extraction, binarization, energy computation and MCH fit are all correct.**

## Issues found
1. **Excluded reaches were not removed.** 157/1249 analysed reaches (12.6%) have `ReachS.exclude = 1`, up to 28 of 72 in 210608, 19 of 64 in 210515 and 19 of 60 in 210620. They have lower `qm` (median 0.58 vs 1.0). The bin10_spike files were produced by an organizeData version that did not export the flag. **Decision (user): drop them, refit, report both versions.**
2. **The static PL fit is unreliable as a generative model in the strongly coupled 2022 sessions.** Its parameters are close to the MCH fit (r ≈ 0.91–0.99), but its model moments are far off (mean activity off by up to 1.7 on the ±1 scale). This happens even with exact enumeration (220516, N = 19), so it is not a sampling problem. These populations sit where model moments are very sensitive to small parameter changes.
   - **Unaffected:** the held-out conditional likelihoods of the time-varying-field model (Fig. 5).
   - **Affected:** any thermodynamic quantity computed from PL parameters (entropy, multi-information, susceptibility; plan step 3). That step must use moment-matched parameters, e.g. PL followed by Boltzmann-learning refinement.
   - **Possibly interesting in itself:** the 2022 populations being close to an ordered regime bears on the susceptibility hypothesis.
3. **Spike counts are binarized:** 21% of non-empty 10-ms bins contain ≥ 2 spikes.
4. **Unused source files:** 210414_WT2 (old format, wild-type), 220519_DJC000 and 220520_DJC002 (no ReachS.stim). Session IDs in the pipeline use the date only. That is unambiguous for the 21 used, but 220519 and 220520 each have a second mouse in the source folder.
5. **Bookkeeping check to redo:** my comparison of per_reach_state `x` used the wrong kinematic column (the last three columns are scaled −1..1). All other columns matched exactly.

## What the raw files add (now available)
- `ReachS.stimtime`: the laser time for every stimulated reach (400/400), a median 3.9 ms after the 1-cm crossing on the 10-ms grid. The stimulation test can use true trigger times.
- `ReachS.filt_kin` / `real_kin`: unscaled hand position (cm, about 120 Hz). The 1-cm crossing sits at window index 400 (IQR 397–401); median peak outward x is 2.0 cm.
- `cellData.channel`, `depth`, `CV2`, `MAD`, `fr`: unit positions for the spatial analysis.
