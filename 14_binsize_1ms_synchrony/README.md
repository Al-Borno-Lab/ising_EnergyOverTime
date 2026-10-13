# E14 · Bin-size sweep and 1-ms synchrony (Nashef reconciliation)

Status: see `experiment.json` · Owner: unassigned · Started: 2026-10-12

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> Motivation: `docs/literature/LITERATURE_NOVELTY_REPORT.md` (P2 verdict) and `METHODS_COMPARISON_REPORT.md` §5 (PR #5).
> Nashef, Spindle, Calame & Person (2023 bioRxiv, PMC10370034; a 2026 *Nat Commun* version is cited by one agent but
> **unverified**) report that 1-ms Purkinje-cell synchrony rises during the reach, in this same lab's task. Our 10-ms
> analysis finds that coupling J does not detectably change (A8).

## Question
Is "synchrony rises during the reach" (Nashef) compatible with "coupling J is fixed during the reach" (ours)? Three possible explanations:
- **(a) Timescale:** the change lives at 1–5 ms and is invisible at 10 ms.
- **(b) Operating point:** expressed synchrony changes because firing probabilities move while J stays fixed (E09 H1, E10).
- **(c) Cell selection:** Nashef's effect is in rate-suppressed cells.

## Hypotheses (freeze in a dated pre-registration before running)
- **N1, replication:** on our reaches and cells, Nashef's 1-ms synchrony index (coincidence probability / product of
  single-cell probabilities, with trial-shuffle correction) rises in the reach window vs baseline. Use their window and
  their test as published; report rate-suppressed vs other cells separately.
- **N2, timescale:** refit the time-varying-field model (M1, fixed J) at 1, 5, 10 and 20 ms bins. Test whether
  epoch-specific coupling (or κ in the reach window, J_eff = (1 + κ)J, as validated in E10/E11) improves held-out
  likelihood at 1 and 5 ms when it doesn't at 10 ms.
- **N3, operating point:** with J fixed, the fitted model's *predicted* pairwise synchrony index in the reach window,
  driven only by h(t), accounts for the measured rise in N1 (fraction explained, per session).

Interpretation:
- N1 yes, N3 explains it: the papers agree, and our model explains the mechanism. This is the best outcome for the paper.
- N1 yes, N2 shows κ > 0 at 1–5 ms: coupling does change, but only at fine timescales. Revise A8's wording.
- N1 no: report the discrepancy and check cell selection and windows against Nashef.

Statistics: unit = session (n = 21); two-sided Wilcoxon; BH-FDR over {N1, N2, N3}; cohort reported separately.

## Data
- **Raw spike times:** `cellData/spikeTimes` (also `cellData/Bin1`, 1-ms counts) in Dylan's files,
  `/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_<date>_<mouse>_NPresults_short.mat` (MATLAB v7.3, h5py).
  - Files hold **all sorted units** (e.g. 54 in 210421), so restrict to our Purkinje cells by matching
    `cellID`/`clusterID` to the units in the bin10 files. `08_exclusion_applied/code/reextract.py` already maps raw → bin10
    exactly; extend it.
  - Check before use: rebinning spikeTimes at 10 ms must reproduce `Bin10` exactly.
- Reach alignment: `ReachS` (bin 400 = 1-cm crossing); `exclude == 1` reaches dropped.
- Copy the files under `$HOME` before sbatch (the container can't see `/data001`), and delete the copies when done.
- Bonus (cheap, from the same files): `cellData/depth`/`channel` for a J-vs-distance analysis. Report descriptively.

## Method
- N1: per pair, the synchrony index in 1-ms bins, reach window vs baseline, shuffle-corrected; session mean over pairs.
- N2: spikes binarized per bin size. Spline knots stay fixed in ms. Exact ML for N ≤ 16 (E11 cap), subsets above.
  The 1-ms fits are sparse: check moment matching, and report if they fail.
- N3: simulate from the fitted fixed-J model at 1 ms (or analytically from pairwise marginals) to get the predicted SI(t).
- Compute: the 1-ms fits are the heavy part (700 bins become 7,000 per reach). Smoke-test one session and estimate CPU-hours before the full run.

## How to reproduce
```bash
bash 14_binsize_1ms_synchrony/code/run.sh
```
Planned scripts: `code/extract_spike_times.py`, `code/sync_index.py`, `code/fit_binsizes.py`, `code/stats_e14.py`.

## Outputs
`assets/tables/sync_index_by_session.csv`, `assets/tables/binsize_cv.csv`, `assets/figures/nashef_reconciliation.png`;
raw outputs go in `results/` (gitignored).
