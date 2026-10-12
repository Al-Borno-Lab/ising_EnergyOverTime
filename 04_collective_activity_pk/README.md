## What this answers

This folder tests whether the population-level activity distribution P(K) — the fraction of active neurons at a given moment — carries its own predictive signal around reach-related kinematic events, and whether it aligns with or is distinct from the pairwise J-coupling peaks found elsewhere in the project.

## Scripts

- `code/pk_j_analysis.py` — Computes/compares P(K) under empirical, independent, and Ising models against detected J-coupling peaks near reach onset; no further CLI/docstring detail was extracted from source beyond the script name, so run it directly (`python pk_j_analysis.py`) and check its argument parser for options.

## Key findings

- Across 21 sessions (42 stim 0/1 pairs, window bins 350–475, peak threshold 1.75), J-coupling peaks are statistically associated with "collective" P(K) behavior: χ² = 4.556, df = 1, p = 0.033. The collective-activity criterion (`r_ising > r_independent` and `ising_indep_dist > 0.1`) was met in 75% of peak sessions vs. 36% of no-peak sessions.
- Mechanistically, this arises because the **independent** model fits worse during J-peaks (r_independent 0.890 vs 0.972), not because the Ising fit itself improves (r_ising ~0.96 in both cases) — i.e., peaks reflect failure of the independence assumption.
- A second, related analysis (using `pk_j_summary.csv`, 63 session/stim rows, 39 J-peak vs 24 no-peak, with a stricter collective-classification rule including an R_TOL=0.02 tolerance band) found a much stronger association: 37/39 (94.9%) of J-peak sessions were "collective" vs. 7/24 (29.2%) of no-peak sessions, OR ≈ 44.9, Fisher's exact p = 4.3×10⁻⁸. This uses a different (looser) threshold/tolerance than the chi-squared analysis above, so the two effect sizes are not directly comparable, but both point the same direction.

## Caveats

- Several detected J-peaks (sessions 210423, 210622, 210511) had near-zero `ising_indep_dist`, suggesting these are likely **false-positive** peak detections (noise) rather than genuine collective events — motivating the dual-criterion filter used in the second analysis.
- Session 220515 is flagged as a **genuine, unexplained exception**: it shows strong J peaks and high divergence, yet the independent model outfits Ising on P(K) — possibly driven by unusually strong single-neuron (h-field) drive. This is unresolved.
- The first note's caveats section is truncated in the source material (cuts off after "Caveats:"), so any additional documented limitations from that analysis are not captured here.
- `notes/pk_analysis_jul10/` contains multiple parallel iterations (`pk_analysis`, `pk_analysis_2`, a scipy-prominence peak-detection variant, and an event-outcome breakdown) whose relative status (superseded vs. current) is not specified in the available summaries.