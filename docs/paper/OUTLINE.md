# Paper outline (draft 1, 2026-10-27)

**Framing (Gunnar):** fitting a time-varying-field pairwise maximum-entropy (Ising) model to Purkinje-cell populations during skilled reaching, and what it shows.

**Sources:**
- Evidence package: `evidence/INDEX.md` (PR #11); figure IDs below refer to it.
- Claims log: `RESULTS_LOG_FOR_PAPER.md`.
- Literature: `docs/literature/LITERATURE_NOVELTY_REPORT.md`.

**Scope rule:** nothing new goes in except E14 (one paragraph, either outcome) unless a co-author or reviewer asks.

## Working title
"Fixed coupling, moving drive: a population model of Purkinje-cell activity during reaching"

## Abstract (skeleton, 5 sentences)
1. **Problem.** Purkinje-cell (PC) coordination during movement has been studied pair by pair, with conflicting conclusions about whether it changes with behaviour.
2. **Approach.** We fit a pairwise maximum-entropy model with time-varying fields (shared reach-locked drive) and fixed couplings to Neuropixels recordings of PC populations (21 sessions, 6 mice), and validate it on simulated data.
3. **Finding: coupling.** Couplings improve held-out prediction beyond time-varying rates in every session, including at rest. They make cells fall silent together about 20% more often than their rates predict. We find no evidence that they change during the reach.
4. **Finding: drive.** The reach is written onto the population as a shared drive that polarizes the cells at peak acceleration, consistently across sessions and not explained by the mean rate. Fixed couplings do not gain influence when the cells polarize.
5. **Meaning.** A stable coupling network and a precisely timed shared drive describe PC population activity; a stationary model would wrongly attribute the movement to rising coupling.

## Main figures
| Fig | Message | Panels (evidence IDs) | Key numbers | Status |
|---|---|---|---|---|
| **1** | The model and why it is trustworthy | 1/01 walkthrough (fix title: 67 splines); 1/02 synthetic validation; 1/05 key-measure validation; 1/04 landmarks | J recovery r = 0.98; gain only when J is real (+0.107 vs ~0); 40/40 sign-correct | Established |
| **2** | Coordination beyond rates, present at rest | 2/04 P(K) difference; 2/02 couplings beat rate-matched surrogates; 2/03 held-out gain over time; S6/01 exact held-out M1 vs M0 | 63/63 above null; 21/21 sessions; exact 21/21, p = 9.5e-7; rest 19/21 | Established (A2, A3, A5, A7) |
| **3** | The reach is carried by fields; couplings do not detectably change | 3/09 absolute energy + rate (M1 vs stationary); 2/03c decomposition; 3/03 stationary model misleads (side by side); A8 CI | Field term 20/21, p_FDR = 2e-5; coupling term ~20× smaller change; A8 −15% (CI −28 to +93%) | Established; hedged wording for A8 |
| **4** | Peak acceleration = population polarization, not mean rate | 4/01 E12 T1/T2; 4/05 field term vs polarization index vs rate; **4/07 Calame reconciliation**; S2/01 shuffle controls (supplement or inset) | 65% vs 14% rest; 50 vs 330 ms; field term 20/21, survives regressing out the rate (20/21), rate 13/21; r = 0.98 vs polarization | Robust (B13) + exploratory checks |
| **5** | What fixed couplings do: joint silences, but no gain in influence | E15 pause time course; 5/01 E10 H2/H4 | Pauses above rates at rest 21/21 (+20%); no extra pause at peak acceleration (12/21); E10 multi-information −0.001, 15/21 negative | E15 descriptive; E10 pre-registered null |

## Supplementary
- **S2:** shuffle controls on the energy time series.
- **S3:** 1-ms refit.
- **E14:** Nashef reconciliation (pending).
- **E11:** stimulation, inconclusive. Short, or omit.
- **S4:** entropy across blocks and baseline-referenced surprise, with the drift caveat. Optional; likely omit.
- **S5:** per-reach locking (low power). Optional.
- **Methods figures:** 1/03 pipeline verification; 5/02 and 5/03 synthetic amendments.

## Discussion points
1. **Relation to Calame et al. 2023 (same recordings).** They first reported the population event at peak acceleration, from pooled rates. We reproduce the pooled step. We add a per-session, pattern-level signal (who goes up and who goes down) that the rate does not explain (Fig 4).
2. **Relation to Nashef et al. (same recordings).** They report rising 1-ms synchrony, while we find no detectable 10-ms coupling change. Reconcile by timescale and cell selection (E14). Synthetic note from E14: shared trial-to-trial gain inflates shuffle-corrected synchrony indices strongly while the model's coupling scale is barely affected. Wording: "compatible with", never "contradicts".
3. **Methodological point.** A stationary Ising model on these data would report coupling rising during the reach (B2). Separating drive from coupling removes that artefact (cf. Tyrcha 2013; Ferrari 2018).
4. **Biological reading.** Effective couplings most likely reflect shared input, not PC–PC synapses. Joint silences matter for the cerebellar nuclei (Person & Raman 2012).
5. **Limitations.**
   - 10-ms binarization.
   - Pseudo-likelihood for N > 20.
   - A8 is not powered to exclude moderate changes.
   - The 2022 cohort carries several secondary effects and is where pseudo-likelihood couplings are least reliable.
   - Endpoint and peak-acceleration jitter is low.

## Methods sections
1. Data, exclusions and landmarks (B1).
2. Model: B-spline fields (67, 50-ms knots near the reach), symmetric J, pseudo-likelihood fitting, exact likelihood for N ≤ 20.
3. Validation: synthetic tests, pipeline reproduction, surrogates.
4. Held-out evaluation and nulls: trial shuffle, 200-ms lag, time shifts.
5. Energy decomposition; E12 signals and polarization index.
6. E10 multi-information / χ_J; E15 pause events.
7. Statistics: session as the unit, Wilcoxon, BH-FDR, Hodges–Lehmann with bootstrap CIs, cohort reporting, pre-registrations and amendments.

## Before submission (short list)
- [ ] E14: read or write as a limitation (null test running, job 4590182).
- [ ] Sensitivity without 210511 unit 11 (duplicated unit) for the Fig 2–4 numbers.
- [ ] Ask Abigail Person: which Nashef version to cite (2023 bioRxiv vs 2026 *Nat Commun*, unverified); co-authorship and data statement.
- [ ] Merge PRs #11 (evidence) and #12 (E15) after spot-check and reproduction.
- [ ] Regenerate the walkthrough figure with the correct title (67 splines).
- [ ] Words to avoid: "first to show PCs are coupled", "contradicts Nashef", "mean rate is unchanged", "criticality".
