# Evidence package: fitting the time-varying-field Ising model to Purkinje cells

Built 2026-10-27. Figures come from the repository and experiment branches, plus exploratory analyses regenerated on 2026-10-27 by `evidence/src/` (branch `analysis/evidence-package`). Pairwise maximum-entropy (Ising) model with time-varying fields h_i(t) (B-splines) and fixed couplings J; 21 sessions, 6 mice, 10-ms bins, 63 session x stimulus conditions. The authoritative claim list is `ising_wt_protocol/RESULTS_LOG_FOR_PAPER.md` (origin/main, updated 2026-10-13); statuses below follow it.

**Status key.** established = log section A; robust = log section B; preregistered-null = preregistered test not supported; inconclusive = preregistered outcome (c); exploratory = not preregistered, not in the log; withdrawn = log section D or retracted.

**PDF key.** "vector" = the PDF was copied from the source (or split page-by-page from a source PDF). "rasterized" = no source PDF existed, so the PNG was wrapped in a PDF at full resolution with its dpi preserved (img2pdf).

**Combined file.** `ALL_FIGURES.pdf` h## Regenerated material (2026-10-27)

The first build of this package (earlier the same day) found the local scratch folders that held the exploratory figures empty, so folders S2-S6 and several panels were missing. They have now been regenerated from saved cluster outputs and new cluster runs (SLURM array 4548957, one job per session, for absolute energy, signal overlays, field-dip shuffles and burster/pauser timing):
- `evidence/src/make_figures.py`: all regenerated figures and tables; `regenerated_stats.json` holds every number quoted from them.
- `evidence/src/m0_vs_m1.py`: S6.
- `evidence/src/regen_session.py`: the per-session cluster job.
- `evidence/src/package/`: this INDEX, `ALL_FIGURES.pdf` and the zip.

Figures marked "regenerated" were redrawn by new code, not copied from the lost files. Their numbers were checked against the values reported before the loss; all match except where noted under **Corrections**. Large intermediate files (`.npz`) are kept outside git, in `code/evidence_data/`.
