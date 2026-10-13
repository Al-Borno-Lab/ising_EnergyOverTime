# E12 · Field-term dip vs firing rate as a marker of peak acceleration

Status: see `experiment.json` · Owner: unassigned · Started: 2026-10-12

> This file is the **experiment write-up**: what we ask and how. Results go in `RESULTS.md`, never here.
> Background: results F1 (field-energy dip at peak acceleration, E09) and B6 (firing rate is more
> movement-modulated than stationary-model Ising quantities, E07). Preliminary figure:
> `assets/figures/prelim_field_vs_rate_trough.png` (exploratory, post hoc, see "Prior look").

## Question
Does the time-varying **field term** of the population energy, −Σᵢ hᵢ(t)sᵢ, mark peak acceleration more reliably than
**firing rate**? And is that advantage more than what a plain readout of individual neurons' rates already gives?

The field term is a weighted combination of single-neuron rates, so this is a claim about *population rate patterns*,
not about coupling. The paper-relevant version is: "a population signal locks to peak acceleration that the
population-average rate misses, because neurons change in both directions." E12 must say whether the Ising field
term adds anything beyond a simple decoder on the same rates.

### Second question: how does our method deviate from standard methods?
**Key insight (2026-10-12).** For an uncoupled cell, hᵢ = atanh(mᵢ) with mᵢ = 2pᵢ − 1, so the cell's mean field term is
−mᵢ·atanh(mᵢ). That is ≤ 0, symmetric in the direction of change, and more negative the further a cell moves from 50/50.
**The field term is therefore approximately a population polarization index.** That would explain both why it beats the
average rate (no cancellation between bursters and pausers) and why its dip coincides with H1 (cells move away from 0.5).
Model-free polarization measures will probably reproduce most of the dip. E12 must find out **where the model's output
differs from standard methods**, and must not assume that it does.

Candidate sources of genuine uniqueness, to be tested:
1. **Coupling-corrected fields:** M1 fields are drives *after* removing what the neighbours' coupling explains. M1's field term vs M0's (h = atanh m exactly) isolates the model's own contribution.
2. **One framework linking the dip to coupling influence:** the same polarization lowers I/χ_J at fixed J (P4, E10). Rate methods can find the dip but can't quantify its effect on coupling influence.
3. **Correct attribution of the energy split:** stationary models put the reach energy drop in J, time-varying models put it in h (note under B2).

## Prior look (exploratory, not a finding)
Computed post hoc on 2026-10-12 from `09_.../assets/tables/field_term_trough_timing.csv` (63 conditions, session-level n = 21):
- Trough within 200 ms before peak velocity: field 60% vs 14% in rest controls (p = 0.0002); average rate 32% vs 12% (p = 0.13).
- Trough depth relative to its own rest control: field 4.7× vs rate 2.0× (field higher in 16/21, p = 0.12).
Because these numbers were already seen, E12's tests must be pre-registered as written below and reported in full,
including the depth comparison that was not significant.

## Hypotheses and decision rule (freeze in a dated pre-registration before running)
Signals compared (all from the same reaches, 10-ms bins, exclusions applied):
1. **FT**: the field term −Σhᵢ(t)sᵢ from the time-varying-field model (E07/E09 fits);
2. **FR**: the population-average spike probability;
3. **|ΔFR|**: the mean absolute single-neuron rate change, a rate-only measure that doesn't cancel opposite changes;
4. **DEC**: a cross-validated linear decoder of "time to peak acceleration" (or a peak-acceleration classifier) from single-neuron rates, trained on held-out reaches;
5. **POL**: model-free polarization indices, Σᵢ|pᵢ(t) − pᵢ,base| and Σᵢ mᵢ(t)·atanh(mᵢ(t)), from smoothed PSTHs (no Ising fit).
6. **FT0**: the field term from the time-varying *independent* model M0 (h = atanh m), the in-model version of POL.
7. **TRAJ**: standard population methods: PCA and demixed-PCA trajectory speed ‖dx/dt‖ and distance from baseline; GPFA latent speed (Yu et al. 2009) if feasible at these N.
8. **POOL**: Calame et al. 2023's pooled, velocity-binned population rate, re-run on our sessions. Calame report a sharp pooled-rate drop at peak acceleration that scales with velocity, so our wording must not claim the mean rate is flat (see `docs/literature/LITERATURE_NOVELTY_REPORT.md`, P3).


Primary tests (session-level, n = 21, two-sided Wilcoxon, BH-FDR across the family):
- **T1, timing:** the fraction of troughs/extrema in [−200, 0] ms before peak velocity minus the same in rest control windows (−1.5/−2/−2.5 s), compared FT vs FR and FT vs |ΔFR|.
- **T2, precision:** median |extremum time − peak acceleration| across conditions, compared FT vs FR, FT vs |ΔFR|, and FT vs DEC.
- **T3, depth:** extremum depth / rest-control depth, compared FT vs FR.

Key contrasts:
- **U1 (model contribution): FT (M1) vs FT0 (M0)** on timing (T1) and precision (T2). This is the only test that isolates the coupling correction.
- **U2:** FT vs POL and TRAJ, i.e. whether standard methods find the same acceleration timing.

Interpretation:
- FT ≈ FT0 ≈ POL: the dip is polarization (a rate phenomenon). The paper's contribution becomes the link to coupling influence (P4), not detecting the dip. **This is the expected outcome.**
- FT beats FT0 and POL: coupling-corrected fields carry extra timing information. A clean uniqueness claim.
- TRAJ matches FT: standard population methods see it too. Say so, and cite.
- FT beats FR but not |ΔFR| or DEC: the effect is "bidirectional rate changes cancel in the average," with no added value from the Ising fields.
- FT beats all: the model's field weighting adds information. Report this as a methods advantage.

Report per cohort (2021 vs 2022), and treat mouse as a random effect.

## Data
- Note: the script that produced `field_term_trough_timing.csv` was not preserved in the handoff, and whether it used exclusion-applied fits is unknown. E12 recomputes everything from spikes.
- Primary: E08 exclusion-applied spikes, `spikes_v2.npz` with `exclude_k == 0` (cluster path in the E10/E11 READMEs), and phase landmarks `09_.../assets/tables/phase_landmarks.csv`.
- Fields hᵢ(t) from the time-varying-field model, M1 in `09_.../code/tvfield_model.py`. Refit if the stored fits aren't the exclusion-applied ones.
- Sensitivity: all reaches; sessions with N ≥ 5.

## Method
- Extremum search window and control windows exactly as in the results log (section F): the same search, shifted −1.5/−2/−2.5 s.
- Each signal is z-scored against its own baseline (−1,000 to −400 ms) before computing depth.
- DEC: 5-fold CV over reaches (the same folds as E07), with ridge regularization chosen by nested CV. Report held-out performance.
- Mostly local compute: no new Ising fits unless the exclusion-applied fields are missing.

## How to reproduce
```bash
bash 12_field_vs_rate_indicator/code/run.sh
```
- `code/make_pi_figure.py`: preliminary exploratory figure from the E09 table (done 2026-10-12).
- `code/compute_signals.py`, `code/decoder.py`, `code/stats_e12.py`: planned.

## Outputs
- `assets/figures/prelim_field_vs_rate_trough.{png,pdf}`: exploratory figure, labelled as such.
- `results/` (gitignored); final figures and tables go in `assets/`.
