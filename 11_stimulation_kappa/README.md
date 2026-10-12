# E11 · Stimulation model: does optogenetic perturbation change coupling (kappa)?

Status: see `experiment.json` · Owner: unassigned · Started: 2026-10-12

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> Background: [HANDOFF](../09_claude_science_handoff/HANDOFF_CLAUDE_CODE.md) §9.4 and §10;
> [PAPER_ANALYSIS_PLAN.md](../09_claude_science_handoff/reports/PAPER_ANALYSIS_PLAN.md) steps 0, 7 and 10;
> results-log entry C ("Stimulation blocks, preliminary").

## Question
When closed-loop optogenetic stimulation is delivered mid-reach (block `stim_1`), does the **coupling** between
Purkinje cells change, or only their firing rates? We measure this as a single coupling scale κ in the pulse window
(J_eff = (1 + κ)·J). Block-specific time-varying fields absorb the rates, so a rate change can't masquerade as a coupling change.

Why it matters: this is the only causal manipulation in the dataset. E07 established that coupling exists beyond
reach-locked rates and doesn't detectably change during natural reaches. E11 asks whether it changes under direct
perturbation. Either answer is paper-relevant: "perturbation changes interaction" or "coupling is invariant even to
perturbation; stimulation acts through rates."

## Prior evidence (not a finding)
In per-block stationary fits, the coupling term around peak velocity is above the last 5 baseline reaches from the first
stimulated reach (+0.93 baseline-SD, 16/21, p = 0.009) through the block and into early washout. **Nothing survives FDR**,
and the stationary model is known to leak rate changes into J (results log, note under B2). E11 is the proper test.

## Hypotheses and decision rule (to be frozen in a pre-registration before real-data fits)
Prediction from PAPER_ANALYSIS_PLAN step 10: extra mossy-fiber drive pushes rates away from 0.5, so **coupling influence
(I, χ_J) falls in the pulse window while κ ≈ 0**.
- **S1 (primary):** κ_pulse − κ_control, tested with a two-sided Wilcoxon over sessions (n = 21). Concluding "coupling is
  invariant" requires an equivalence test: the CI on κ_pulse − κ_control must exclude |κ| ≥ 0.3. The bound will be fixed
  in the pre-registration. p > 0.05 alone is not enough.
- **S2:** Δ multi-information per neuron and Δχ_J/N in the pulse window vs the matched baseline window (fixed J, exact `thermo()`).
- **S3 (exploratory):** κ in early vs late stimulated reaches (adaptation) and in washout (persistence).

BH-FDR over {S1, S2}. Report cohort (2021 vs 2022) separately, and treat mouse as a random effect.

## Data
- 21 sessions × 3 blocks (stim_0 baseline, stim_1 stimulation, stim_2 washout), 10-ms bins, binarized, rows 100:800.
  Primary analysis drops `exclude == 1` reaches; using all reaches is the sensitivity analysis.
- **Spikes:** `spikes_v2.npz` (on the cluster in `09_claude_science_handoff/data/`, gitignored), filtered with `exclude_k == 0`.
- **True laser times:** `ReachS.stimtime` in Dylan's raw files
  `/data001/projects/enserrog/dylan/AbigailUpdated/Dylan_<date>_<mouse>_NPresults_short.mat` (MATLAB v7.3, read with h5py).
  - They are not in spikes_v2.npz yet; step 1 adds them.
  - Copy the files under `$HOME` first, because the container can't see `/data001` from sbatch.
  - Expected: the pulse starts about 1 bin after bin 400 (9.5-ms latency, 50-ms train at 100 Hz, 2-ms pulses).
- **Open question for the lab:** are all 6 mice opsin-positive? The plan says all received AAV-ChR2 in the pontine
  nuclei, with red-light sessions as controls, but the handoff lists this as unconfirmed. Any opsin-negative or red-light
  sessions are a negative control (expect κ ≈ 0 and no rate change). Confirm before the real-data analysis is final.

## Method
A joint model with **one fit per session across all three blocks**:
- **Shared J** across all blocks and reaches (the baseline coupling).
- **Block-specific time-varying fields** h_i^b(t): cubic B-splines as in `tvfield_model.bspline_basis(400)`, one set per
  block, so stimulation-driven rate changes go into h.
- **Per-group/window intercepts:** a small offset per (block, window) group, so a uniform shift isn't attributed to κ.
- **κ:** a scalar coupling scale, J_eff = (1 + κ)·J, fitted separately in two windows:
  - the **pulse window**: stim_1 reaches aligned to the true laser onset. The width is fixed in the pre-registration
    (e.g. onset to +60 ms).
  - a **control window**: the same reach phase in stim_0 reaches, aligned to bin 400 + median latency.
- **Fitting:** exact maximum likelihood for N ≤ 20 (extend `09_.../code/thermo_model.py`), and 10 random 16-neuron subsets
  for N > 20 (as in E10, amendment 1).
- **Null:** 20 trial-shuffle refits per session.

## Steps
1. **Rebuild inputs.** Extract `stimtime` per reach from the raw files and match it to `reachnum_k` in spikes_v2. Check
   that onset falls at about bin 400–401. Save as `assets/tables/laser_onsets.csv`.
2. **Rewrite the lost code:**
   - `code/stimfit_model.py`: the joint model;
   - `code/validate_stimfit.py`: synthetic validation;
   - `code/run_stimfit.py`: the real-data run.

   Unit-test the likelihood against brute force, as was done for `thermo_model.py`.
3. **Synthetic validation (method only, labelled as such).** The model must:
   - recover κ = 0.5 (the previous build gave 0.50–0.53);
   - give κ ≈ 0 when stimulation changes only rates;
   - give no false positive when rates change only in the pulse window.

   This must pass before any real data is touched.
4. **Pre-register** S1–S3, the windows and the equivalence bound in a dated document in this folder, and commit it.
5. **Run on real data on the cluster** via sbatch. Smoke-test one session first, and log every run.
6. **Statistics and write-up** in `RESULTS.md`, with proposed results-log entries for review.

## How to reproduce
```bash
# from the repo root, on the cluster (never run heavy work on the login node)
bash 11_stimulation_kappa/code/run.sh
```
Scripts in `code/` (none written or run yet):
- `code/run.sh`: entry point; runs steps 1, 3 and 5 in order.
- `code/extract_laser_onsets.py`, `code/stimfit_model.py`, `code/validate_stimfit.py`, `code/run_stimfit.py` (planned).

## Outputs
- `results/` (cluster only, gitignored): raw outputs.
- `assets/figures/`, `assets/tables/` (committed): every figure and table cited in `RESULTS.md`.
