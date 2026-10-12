# Result: coupling (J) modulation without firing-rate modulation is rare

**Question.** Are there conditions in which population firing rate stays mostly flat over the reach, yet the coupling term shows a peak or the recording shows strong coupling structure (Ising fits better than the independent model)?

**Answer.** Yes, but rarely. Across 63 session × stimulus conditions, movement modulation of J and of firing rate are strongly correlated (Spearman ρ = 0.74, p = 3e-12). One condition, **210511 stim 1**, clearly shows J modulated with a flat firing rate. A second, 210619 stim 2, partially does.

![J vs firing-rate modulation]({{artifact:art_a9388ff0-e381-4deb-a2d2-4bf4f2559899}})

![210511 stim 1]({{artifact:art_92c2b1b8-a8d4-4aad-94f3-f1aaf538c8dc}})

## Method
- **Movement modulation:** the peak |z| of the 5-sample-smoothed trial-mean trace within peak acceleration −50/+100 samples. The z-score is relative to the pre-reach baseline (peak velocity −200 to −40). Values come from `arbitration_per_condition.csv`.
- **Flat firing rate:** the bottom third of firing-rate modulation (peak |z| ≤ 6.83; 21 conditions). Candidates were then checked visually on the per-condition six-panel figures.
- **Ising advantage:** from the cluster's `model_quality_summary_P_K.csv`, computed as the Jensen-Shannon distance between the data P(K) and the independent-model P(K), divided by the distance between the data P(K) and the Ising-sampled P(K). This file exists for 57 of 63 conditions, 56 of which matched the phase-analysis conditions.

## Results
| Condition | N | FR mod. \|z\| | J mod. \|z\| | Ising / independent | Verdict |
|---|---|---|---|---|---|
| 210511 stim 1 | 12 | 2.8 | 5.5 | 2.7× | Clear: J peaks at +13 samples from peak velocity with an energy trough at the same time; FR stays near noise level |
| 210619 stim 2 | 31 | 6.2 | 7.7 | 2.2× | Partial: sustained J step-up before the reach, only a small FR bump |
| 210608 stim 1 | 21 | 4.9 | 6.9 | 3.0× | Not a dissociation: the J peak coincides with an FR bump |
| 220520 stim 0 | 27 | 5.0 | 7.3 | n/a | Misclassified by the score: FR drops clearly after the reach |

- The Ising model is closer to the data P(K) than the independent model in 55/56 conditions, including all 20 flat-FR conditions that have the comparison.
- The Ising advantage is smaller in the flat-FR group (median JS gain 0.031 vs 0.052 in the rest) and increases with FR modulation (ρ = 0.41, p = 0.002). The strongest coupling structure occurs where firing rate also moves most.

## Limits
- The modulation score has a noise floor of roughly |z| ≈ 3. A noisy baseline can make a modulated FR look flat, as with 220520, so visual confirmation is required.
- The Ising-vs-independent comparison is computed once over the whole recording and in-sample. It shows coupling structure overall, not coupling specifically at the moments when FR is flat. A time-resolved comparison (sliding window, or around J peaks) is needed to make that claim.
- This is one clear condition out of 63 and should be reported as an example, not a population effect.
