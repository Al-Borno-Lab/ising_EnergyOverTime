
## Corrections and verification status (all numbers checked 2026-10-27)
1. **B-spline count:** 67 (code). The `tvfield_walkthrough.png` title says 66; the title is off by one.
2. **"Sign correct 40/40":** verified. It is scenarios A and B of the H0 synthetic test (20 + 20); see 1/05 for every scenario.
3. **"Pseudo-likelihood vs exact gain rho = 0.999":** NOT found in any saved table. Do not cite; the saved result is "exact and PL agree in 38/39 conditions".
4. **Held-out coupling gain:** 21/21 sessions, 63/63 above the shuffle null 95th percentile (A7), median +0.26 nats/bin. At rest (baseline phase): 20/21.
5. **200-ms lagged control:** recovers about 1/5 (0.051 vs 0.259 nats/bin, tvfield_phase_summary) or about 1/4 (0.05 vs 0.18, RESULTS_LOG). These are two different summaries; say "a small fraction (1/5 to 1/4)".
6. **"20x smaller":** compares the time-varying model's *coupling term* with its *field term* (about -0.01 vs -0.17 per neuron). It is not a comparison with the stationary model.
7. **Rates vs couplings (13 sessions):** rates-only beats couplings-only during the reach in 11/13 sessions, p = 0.15 (not significant). Time-varying Ising beats time-varying independent 12/13, p = 0.0005: HANDOFF section 7 and tvfield_phase_summary.csv (exact LL, p_FDR = 0.0005).
8. **E12 r = 0.98 / -0.07:** reproduced (4/05). These are exploratory trace correlations, not E12 preregistered outputs.
9. **E10 H3b:** the verdict is "no J change", but kappa median +0.19 (13/21, p = 0.055) and the 2022 cohort alone is 6/6 positive (p = 0.031).
10. **E10 convergence:** 54/99 non-converged real-fit tasks in 2022, 19/180 in 2021.
11. **P(K) 21/26:** reproduced (2/04).
12. **S6 M1 vs M0:** reproduced (21/21, 63/63, median 0.0115 bits/neuron/bin, p = 9.5e-7; rest 19/21; reach 20/21). The relative gain is 1.6% as the median of per-condition ratios; the earlier "1.3%" used a different summary.
13. **Exploratory numbers reproduced from regenerated code:**
    - S2: 18-19/21 vs 6-7/21; coupling 21/21, 19/21, 6/21.
    - S4: 17/21 (p = 0.0004); trend 18/21 (p = 0.006); washout 19/21.
    - S5: rho 0.09, 13/21 (p = 0.04); re-alignment p = 0.37; jitter 54 ms.
14. **Verified exactly, unchanged:**
    - synthetic +0.107 / about 0 / about 0, with r_J = 0.978;
    - |J| 63/63;
    - field 20/21, p_FDR = 2e-5;
    - A8 -15% (CI -28 to +93%);
    - E12: 65% vs 14%; 50 vs 330 ms; p_BH = 0.024;
    - E10: H2 -0.001, 15/21, p = 0.29; 2021 13/15, p = 0.015; H3a 0.34 and 13/21; H4a 0.70; H1 19/21;
    - E11: -0.266 [-0.469, -0.022], p_FDR 0.13; S2 p = 0.79; 2021 -0.44; 2022 +0.17.

## Still not recoverable
- The curated `paper_figures_by_point/` panels (1a-4e) and the old `findings_png/` set: repository equivalents and the split Sept-30 PDF pages (X_) are used instead.
- Data flag (not a missing file): session 210511 unit 11 looks like a duplicated unit at 1 ms (74% of active 1-ms bins have >= 2 spikes). Check spike sorting before final analyses.

## Repository figures not included (available in `09_claude_science_handoff/assets/figures/`)
strong_session_profile, session_diagnostics, example_flatFR_210608_stim1, example_flatFR_210619_stim2, example_flatFR_220520_stim0, example_phase_timing_220517_stim1. fig2_collective, fig3_not_artefact, fig4_tvfield_validation, fig5_coupling_beyond_rate, fig10_FR_most_modulated are raster pages of the Sept-30 findings PDF and are covered by the split pages in X_.
