# E10 · Operating point: amendment checks and H2-H4 on real data: results

> The **results write-up**. Only claims backed by a file in `assets/` or a logged run.
> Each claim cites its asset and the run (see `runs/RUN_LOG.md`).

## Summary (plain language, 3–5 sentences)

## Results
| # | Result | Key numbers | Asset | Run |
|---|---|---|---|---|

## Caveats and open issues
- **Neuron-level internal contrast lost (H4b).** In synthetic validation (step 1, `assets/tables/step1_summary.csv`, `assets/figures/step1_synthetic_amendments.png`) the trial-shuffle null reproduced the neuron-level relation between change in distance from p = 0.5 and change in influence (rho_real ~ rho_null ~ -0.6 to -0.8 in scenario E), so rho_real - rho_null cannot detect the mechanism. H4b is therefore exploratory and descriptive only (pre-registration amendment 6, written before any real-data fit). The mechanism is tested only across conditions/sessions (H4a and the session-level correlation of delta p(1-p) with delta I/N).

## Entries for RESULTS_LOG_FOR_PAPER.md
Which log IDs (A/B/C/D) this adds or changes. Copy them into the top-level log only when sure.
