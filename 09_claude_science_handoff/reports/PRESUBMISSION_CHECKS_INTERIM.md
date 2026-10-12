# Pre-submission checks: interim state (paused 2026-09-26)

Plan: close the six open items in RESULTS_LOG_FOR_PAPER.md, Section E.

## Done
- Solver: refits in the project Singularity container (inverse-ising-arm-2.sif) reproduce the stored parameters (J r >= 0.998, h r = 1.0, slope 0.98-1.02; 3 conditions).
- Item 6, bin width: dt = 10 ms in all 1,088 matrices across the 9 raw .mat files. 1 sample = 10 ms.
- Item 2, surrogate refit: 100 rate-preserving shuffles per condition, refitted with the pipeline's coniii MCH. Real mean |J| is above all 100 surrogates in 63/63 conditions (median 4.6x, min z = 3.6). Net coupling sum is above surrogates in 62/63; the exception is 210622 stim 2, N = 3. Files: surrogate_refit_results.csv, surrogate_refit_J.png.

## Interim (analysed, not yet written up)
- Item 1, co-silence: the co-active and co-silent pair contributions to the change in the J term are not individually significant. They track the firing-rate change in opposite directions (rho = +0.89 and -0.91). The movement-related J rise is carried mainly by fewer discordant pairs (p_FDR <= 0.003 in every phase).
- Item 1, follow-up (PRELIMINARY): splitting the trial-averaged J term into a part predicted from per-neuron rates alone and a residual across-reach covariance part, only a median 34% of the movement-window change comes from covariance, and neither part is significant on its own at the session level. The J-term rise may be largely rate-driven through the fitted couplings. The rate-only prediction has a 1/R self-term bias that still needs a leave-one-reach-out correction before any conclusion is drawn.

## Pending
- Item 3: cross-validated Ising vs. independent log-likelihood per bin. Finished on the cluster; output in hpc/3901ca2c-.../timeresolved_llr.csv.gz, not yet analysed.
- Item 4: dedup refit of 210620/220515. Finished on the cluster; hpc/3901ca2c-.../dedup_units.csv and dedup_per_bin.csv.gz, not yet analysed.
- Item 5: the regeneration job (597e3c04) writes to 07_presubmission_checks/regenerated/ on math-alderaan; outputs not yet collected.
- Re-render the biased figures; write OPEN_ITEMS_RESOLUTION.md; update RESULTS_LOG_FOR_PAPER.md.
