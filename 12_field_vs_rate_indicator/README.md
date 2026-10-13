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
5. *(optional)* **PC1**: the leading principal component of single-neuron rate trajectories.

Primary tests (session-level, n = 21, two-sided Wilcoxon, BH-FDR across the family):
- **T1, timing:** the fraction of troughs/extrema in [−200, 0] ms before peak velocity minus the same in rest control windows (−1.5/−2/−2.5 s), compared FT vs FR and FT vs |ΔFR|.
- **T2, precision:** median |extremum time − peak acceleration| across conditions, compared FT vs FR, FT vs |ΔFR|, and FT vs DEC.
- **T3, depth:** extremum depth / rest-control depth, compared FT vs FR.

Interpretation:
- FT beats FR but not |ΔFR| or DEC: the effect is "bidirectional rate changes cancel in the average," with no added value from the Ising fields.
- FT beats all: the model's field weighting adds information. Report this as a methods advantage.

Report per cohort (2021 vs 2022), and treat mouse as a random effect.

## Data
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
