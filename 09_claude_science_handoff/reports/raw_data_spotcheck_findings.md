# Raw-Data Spot-Check: Independent Recomputation from .mat Files

## Method

For **every raw MATLAB session file present on the cluster** (`/home/enserrog/ising_EnergyOverTime/data/*.mat`,
9 files — the complete raw-data inventory), I independently re-derived the pipeline's per-timestep spike
matrix and per-timestep energy decomposition **from scratch**, without importing or calling any of the
audited project code, and compared the result against the pipeline's own saved outputs
(`per_reach_state.csv`, `h_parameters.csv`, `J_parameters.csv`, `preprocessing_info.csv`).

Independent re-implementation covered:
1. **Raw loading**: read `kinAggrogate.stim_N.n_*` neural count arrays directly via `scipy.io.loadmat`
   (own from-scratch MAT-struct-to-dict converter, not the project's `loadmat` helper).
2. **Truncation**: sliced `[Truncate_Index_Lower : Truncate_Index_Upper, 1:]` per each session's own
   recorded `preprocessing_info.csv` values (100:800 in every session checked).
3. **Binarization**: reproduced `preprocessingSpikes` — `1 * (sliding_window_sum(spikes, bin_size) > 0)` —
   from its docstring/formula only, independently coded.
4. **Spin encoding**: `2*binary - 1` → {-1, +1}.
5. **Energy decomposition**: reproduced `calc_e_with_terms` (`e = -J_term - h_term`) as a plain NumPy
   function (no numba/coniii dependency), fed with the pipeline's own fitted `h_parameters.csv` /
   `J_parameters.csv` for that exact session/stim.
6. Neural reach keys (`n_1`, `n_20`, ... ) were consumed in their native struct-field order per stim
   (some stims number reaches non-contiguously, e.g. `n_20..n_41`) — matching the original
   `load_and_preprocess_data` iteration order exactly, rather than assuming a fixed `1..N` range.

Checked across **all 9 available raw sessions × all 3 stim conditions = 27 of 27 possible combinations**
— full coverage of every raw file that exists on the cluster.

## Results

| Check | Result |
|---|---|
| Spike matrix exact match (raw → binarized) | **27/27 PASS**, bit-exact |
| Energy exact match (`calc_e`) | **27/27 PASS**, max abs diff 1.07e-14 (float64 rounding) |
| J-term exact match | **27/27 PASS** |
| h-term exact match | **27/27 PASS** |

Full per-condition table (dates, neuron/reach counts, match flags): `verification_spotcheck.csv`.

**Conclusion: no bug found in the raw-data → spike-matrix → spin-encoding → energy-decomposition path.**
Every value the pipeline reports in `per_reach_state.csv` (`spike_n*`, `energy`, `j`, `h` columns), for
every session where raw data still exists on the cluster, is reproducible byte-for-byte from the
original `.mat` files using only the documented formulas, independent of the audited code. This is the
most safety-critical stage in the pipeline (it feeds every downstream figure and statistic), and it is
now fully verified correct against ground truth.

## Important secondary finding: raw data missing for 12 of 21 processed sessions

While enumerating raw `.mat` files, I found only **9** raw session files under
`energy_over_time_script/data/` (dates 210421, 210422, 210423, 210425, 210511, 210512, 210514, 210608,
210614 — all 9 now spot-checked above), but **21** session result directories under
`outputs/energy_decomp_Jul_10_svg/*_results/` (those 9 dates **plus** 210515, 210606, 210619, 210620,
210622, 210623, and six 2022 sessions 220515–220520).

A recursive search of the entire `ising_EnergyOverTime` tree for any file (any extension) matching those
12 extra dates turned up **only derived outputs** (PNGs, CSVs already under `outputs/`) — no raw `.mat`,
`.h5`, `.npz`, or other primary data file exists on the cluster for them. This means:

- **12 of the 21 sessions currently included in aggregate figures/statistics cannot be independently
  verified against raw data** with what's on the cluster today — only the pipeline's own derived
  outputs exist for them.
- If the raw files were moved, archived elsewhere, or deleted after processing, this should be resolved
  before the paper reports results from those sessions, since their processing chain is currently
  unauditable from this cluster alone.
- This is **not** evidence of a bug in those 12 sessions — only that they are outside what could be
  checked here. Recommend locating the original raw files (backup, external drive, or whoever provided
  them) and re-running this same spot-check script against them.

## Recommendation
Locate/restore raw data for the 12 sessions with no primary data file on the cluster (210515, 210606,
210619, 210620, 210622, 210623, 220515–220520), then re-run this spot-check (same script, generalized
across sessions) against them for full-pipeline coverage.
