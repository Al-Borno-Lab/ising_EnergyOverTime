# shared/

## What lives here

Cross-cutting code and data used by more than one of the question-specific folders (`01`-`06`). Anything here is a shared dependency or an aggregate output that summarizes across questions rather than belonging to a single one.

## src/ package

Importable package (`import shared.src...` or added to path) with modules used by scripts across the question folders:

- **arbitration.py** — shared arbitration-related utilities (no docstring provided).
- **peak_detection.py** — detects significant peaks, troughs, and level-shifts in 1-D time series within a search window. Public API: `gaussian_smooth` (Gaussian kernel convolution), `local_prominence` (local prominence of a candidate peak), `detect_j_peak` (detects a bump or step in a J-coupling time series), and `detect_signal_extremum` (generalized peak/trough detector). Used wherever a question folder's scripts need to locate events in a time series (e.g. J-coupling or energy traces).
- **processing.py** — shared data-processing utilities (no docstring provided).
- **util.py** — shared general-purpose helper functions (no docstring provided).

## code/ utilities

Standalone scripts for aggregation/reporting, not tied to any single question:

- `aggregate_model_quality_plots.py` — walks a results tree and builds one 4-panel model-quality summary figure per session-group (e.g. begin/mid/full reach).
- `aggregate_transition_analysis.py` — walks a results tree, loads each session's `transition_analysis_stim_<N>.csv`, and runs a (customizable) hook per session, e.g. printing summaries and plotting energy-vs-time with transitions marked.
- `generate_session_report.py` — discovers all PNGs under a results tree and builds a single Markdown (optionally PDF) image report grouped by chart type; feeds the `docs/` galleries.
- `compare_sessions.py` — builds paper-ready side-by-side comparison figures (model-quality and pk-analysis panels) for two chosen session/stim pairs.
- `regen_model_quality_summary.py` — redraws model-quality summary figures with updated formatting from already-saved CSVs, without re-fitting/re-sampling.
- `collect_files.py` — recursively finds files by name or glob pattern under a root and copies them to a destination, optionally preserving subdirectory structure.

## data/ and results/

- `data/critical_values.csv`, `data/energy_kinematics.zip` — shared reference data used by multiple question folders.
- `results/` — cross-session aggregate outputs that don't belong to one question: `outputs_collected/`, `comparisons/`, an older `jun22-addSpikes` per-session snapshot, and a top-level aggregate model-quality PNG.