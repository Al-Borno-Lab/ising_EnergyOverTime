# 01_ising_model_fitting/

## What this answers
How well a pairwise Maximum-Entropy (Ising) model fits Purkinje-cell population spiking during reaching, and what the model's phase-transition structure (energy landscape, critical temperature behavior) looks like across recording sessions.

## Scripts
- [main.py](code/main.py) — End-to-end pipeline for the standard MATLAB data path: loads/preprocesses spikes, fits the Ising model, runs phase-transition and energy analyses, and writes CSVs/plots. Run: `python main.py --matlab_file path/to/file.mat [options]`.
- [main-wells.py](code/main-wells.py) — Same pipeline adapted for Wells-lab data files. Run: `python main-wells.py --wells_file path/to/file --mouse_id MOUSE_ID [options]`.
- [main-wells-old.py](code/main-wells-old.py) — Earlier variant of the Wells-lab pipeline that reads from `aligned_data.json` and filters neurons by cortical/cerebellar layer before fitting. Run: `python main-wells-old.py --mouse_id MOUSE_ID --layers LAYER1 LAYER2 [options]`.
- [model.py](code/model.py) — Core Ising model implementation: energy calculations, Metropolis sampling, and phase-transition analysis routines used by the `main*.py` scripts.
- [analysis.py](code/analysis.py) — Post-fit analysis functions for energy patterns and pairwise correlation structure.
- [visualization.py](code/visualization.py) — Plotting functions for model-quality, phase-transition, and energy-distribution figures.
- [utils.py](code/utils.py) — Shared data loading, preprocessing, and general helper functions used across the other modules.

## Key findings
No dedicated narrative analysis note was found among the reviewed markdown files for this folder, so no key findings can be reported here without inventing content. The `results/` directory contains the underlying quantitative outputs (per-session phase-transition, model-quality, correlation-order, and energy-distribution files across 21 sessions from 2021-04 through 2022-05, plus cross-session aggregates in `model_quality_summary/` and `collected/`), but interpretive conclusions have not yet been written up in note form.

## Caveats
None documented — no analysis notes with caveats or contradictions were found for this folder. Once a narrative note is written, this section should be updated to reflect any stated limitations (e.g., sample-size sensitivity, convergence issues in the Metropolis sampling, or session-to-session heterogeneity) rather than leaving this placeholder.