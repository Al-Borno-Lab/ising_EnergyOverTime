# Evidence package

The paper's framing is fitting the time-varying-field Ising model to Purkinje-cell populations and reporting what it shows. This folder collects every supporting figure, each with a caption and its status.

- `INDEX.md`: one entry per figure (source, axes, pattern, why it matters, key numbers, status), plus corrections and verification notes.
- `ALL_FIGURES.pdf`: every figure in INDEX order, one per page with its caption.
- `figures/`: figures and tables regenerated on 2026-10-27, as PNG + vector PDF + CSV. `regenerated_stats.json` holds every number quoted from them.
- `src/`: code.
  - `regen_session.py`: cluster job, one SLURM task per session. It refits and writes the absolute energy terms, E12 signal overlays, field-dip shuffle controls and burster/pauser timing.
  - `make_figures.py`: all regenerated figures and tables.
  - `m0_vs_m1.py`: S6, exact held-out M1 vs M0.
  - `package/`: builds the full package (`manifest.py` = figure list and captions; `build.py` = copies sources, writes INDEX.md, ALL_FIGURES.pdf and the zip).

Folders named S* and everything marked "exploratory" were not preregistered. Folder X_ holds withdrawn or superseded figures and must not be cited.

## Reproduce
Intermediate cluster outputs (`.npz`, per-session CSVs) are too large for git and live outside the repository, in `code/evidence_data/`:
- `shuffle/`, `onems/`, `entropy/`, `reachlock/`, `realign/`, `regen/`;
- the cluster scripts that produced them, under `cluster_code_*`.

```bash
python evidence/src/make_figures.py ../evidence_data . evidence/figures
python evidence/src/m0_vs_m1.py ../evidence_data evidence/figures
cd evidence/src/package && python build.py ../../../../evidence_package
```
`build.py` also reads the E10 worktree (`exp/10_operating_point_h2_h4`) and the E11 assets (`exp/11_stimulation_kappa`); paths are at the top of `manifest.py`.
