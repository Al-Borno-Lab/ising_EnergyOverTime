# Literature novelty check: Purkinje-cell population coupling, synchrony and encoding during reach

> Agent-generated 2026-10-12 (background research agent). Citations were checked by the agent against Crossref/PubMed/publisher records. **Open discrepancy:** the novelty report cites Nashef, Spindle, Calame & Person 2026, *Nat Commun* 17:7755 (DOI 10.1038/s41467-026-74576-x) as the published version of the 2023 preprint; the methods report and a follow-up search found only the bioRxiv preprint (PMC10370034). Verify before citing.

Prepared 2026-10-12. Builds on `09_claude_science_handoff/reports/literature_review_coupling_methods.md` (the "internal review"). That review covers the inverse-Ising solver literature, the field/coupling split, and general correlation methods. It is not repeated here. This report adds cerebellar physiology, nonstationary and time-varying maxent models, operating-point and criticality work, and perturbation studies. It then sorts claims P1–P5, B6, B10 and the late-epoch observation by novelty.

## 0. Search coverage and the two most important findings

**Coverage.** About 60 web searches and full-text fetches (PubMed/PMC, bioRxiv, arXiv, Crossref, Nature, Cell, eLife, J Neurosci, PNAS, PLoS CB, Phys Rev). The bibliography (Section 5) lists 62 sources, including two carried over from the internal review. Each was checked against a publisher, PMC, arXiv or Crossref record in this session. Full text was read only for Nashef et al. 2023 and 2026, Calame et al. 2021/2023 (local PDF) and Delamare & Ferrari 2022. For the rest, only the abstract or summary was read, so detailed claims about their methods could be incomplete. Not covered: Google Scholar directly (WebSearch only), conference abstracts (SfN/Cosyne), and non-English literature.

**Requested items I could not verify (not cited as fact):** Bell & Grimm 1969 (only cited second-hand, in De Zeeuw & Romano 2022). "Payne et al. 2019" on PC synchrony was not found under that name. I did not search for Lee et al., Tsubota et al. or Proville et al. 2014 (optogenetics), Medina & Lisberger, or Han et al. 2020, so none of them is assessed here.

**Two findings matter more than the rest:**

1. **The same dataset has already been analysed for PC covariance, and the result is now peer-reviewed.** Nashef, Spindle, Calame & Person, *Nature Communications* 17:7755 (2026), "Behavioral correlates of Purkinje cell ensemble covariance", analyses the Calame et al. (2023) Neuropixels recordings (25 sessions, 7 mice, lobule 4/5 and simplex). This is the published version of the 2023 preprint (PMC10370034) that the project already cites. Using 1-ms joint PETHs and cross-correlograms with shift-predictor, shuffle and jitter controls, they report:
   - co-firing rises markedly during reach in reach-suppressed ("cluster 1") and early-facilitated ("cluster 2") PCs;
   - about 43% of cluster-1 pairs show significant zero-lag co-firing 100 ms before endpoint;
   - co-firing tracks deceleration gain and endpoint undershoot while rate does not.

   The paper uses no maximum-entropy, Ising or GLM model and does not analyse the stimulation blocks (neither appears in the full text I read). The paper should cite the 2026 version, and its "no coupling increase" result (P2) must be reconciled with it explicitly (Section 3).
2. **Calame et al. (2023; preprint 2021, same dataset) already describe a population-level event at peak acceleration.** Individual PCs split roughly evenly into bursters and pausers (22 vs 24 of 46 cells). Their broad individual modulations "vanished at the population level" when pooled, leaving a sharp net drop in pooled simple-spike rate timed to peak outward acceleration and scaling with velocity (R² = 0.83). This bears directly on P3: the timing and the cancellation idea are prior work, and their pooled data show a mean-rate drop at the moment where we report the mean rate "barely changes".

**No published pairwise maximum-entropy (Ising) model of Purkinje-cell or cerebellar-cortex spiking turned up**, in three targeted searches plus the citation trails of the cerebellar papers above. That is the most defensible methodological novelty. It is still "not found in my search", not proof of absence.

---

## 1. Summary table

| Claim | Verdict | Closest prior work | What is new here | Confidence in verdict |
|---|---|---|---|---|
| **P1** PC populations show pairwise coupling beyond shared reach-locked rate modulation, present at rest; Ising beats independent models out of sample; \|J\| beats rate-preserving shuffles | **Partly known.** The existence of coupling is known. The model-based, rate-controlled form is new | Heck et al. 2007; de Solages et al. 2008; Wise et al. 2010; Cao et al. 2017; Nashef et al. 2023/2026 (same dataset, 1 ms); Herzfeld et al. 2023; method: Granot-Atedgi et al. 2013, Ferrari et al. 2018, Delamare & Ferrari 2022 | First pairwise max-ent (Ising) description of PC populations, as far as I found. Cross-validated against a *time-varying-rate* independent model, so the coupling is shown to exceed shared reach-locked modulation at 10-ms resolution, at rest and across sessions | Moderate-high that the Ising-on-PC application is new. High that "PCs are correlated" alone will be called known |
| **P2** No evidence coupling strengthens during reach at 10 ms; movement change carried by fields; stationary-model "J rises" is a rate artefact | **Partly known, and it conflicts with the same-dataset paper.** The artefact half is methodologically known. The biological half is contested | For: Herzfeld et al. 2023 (rate code; synchrony small; standard metrics conflate rate and covariance); Tyrcha et al. 2013; Ferrari et al. 2018; Okun et al. 2012; Brody 1999; Ventura et al. 2005. Against: Nashef et al. 2023/2026; Sedaghat-Nejad et al. 2022; Heck et al. 2007; Wagner et al. 2021 (complex spikes); Brown et al. 2024 | Shows, in cerebellum and during movement, that a stationary Ising model turns rate changes into apparent J changes, and that a time-varying-field model removes it (epoch-specific J does not improve held-out fit). Also a 10-ms, population-model counterpart to 1-ms pairwise synchrony | High that the artefact mechanism has precedent. High that reviewers will cite Nashef 2026 as contradicting the biological claim |
| **P3** Field-term energy dips at peak acceleration (20/21) while mean rate barely changes; opposite-direction changes cancel | **Partly known; one detail at risk of contradiction** | Calame et al. 2021/2023 (same data: burster/pauser cancellation, pooled pause at peak acceleration scaling with velocity); Herzfeld et al. 2015 (burst + pause populations jointly encode saccade speed); Fakharian et al. 2025 (PC vectors cancel); Hong et al. 2016 | A per-session, single-model scalar (field energy) that detects the peak-acceleration population event in 20/21 sessions, framed as a change in population *pattern* rather than in mean rate | Moderate. Timing and cancellation are prior. "Mean rate barely changes" must be reconciled with Calame's pooled drop |
| **P4** During acceleration PCs polarize (p(1−p) falls), so coupling influence (multi-information, susceptibility) falls at fixed J | **Novel as an empirical claim about PCs; the ingredients are textbook/known** | de la Rocha et al. 2007 (correlation depends on rate); Cohen & Kohn 2011; Churchland et al. 2010 (variability quench); Tkačik et al. 2015, Mora & Bialek 2011 (operating point); Nonnenmacher et al. 2017, Schwab et al. 2014 (criticality signatures driven by rates and latent inputs); Donner et al. 2017, Gaudreault & Shimazaki 2018 (time-resolved max-ent macroscopic quantities) | Found nothing showing that a motor population moves its operating point during movement, so that fixed couplings matter less while rate drive dominates. Gives a mechanistic reading of B10 | Moderate. Expect a reviewer to say "this follows from the Ising equations"; it needs an empirical null (Section 4) |
| **P5** Does optogenetic perturbation of mossy-fiber input change PC coupling (κ) or only rates? | **Open question. Either answer would be new** | Calame et al. 2023 (stimulation drives both increases (19) and decreases (10) in PC rate, which adapt over the block; rate only); Guo et al. 2021 (pontine stimulation in reach, rate-level recordings); Ferrari et al. 2018 (retinal couplings invariant across stimulus ensembles); Tavoni et al. 2017 (Ising couplings change after learning in about 20% of PFC sessions); Wagner et al. 2021 (olive-feedback perturbation changes complex-spike synchrony) | No study found that measures PC simple-spike correlation or coupling under in-vivo mossy-fiber/pontine optogenetics, nor across an adaptation block | Moderate (absence-of-evidence claim) |
| **B6** Firing rate is more movement-modulated than any Ising quantity | **Known / consistent** | Herzfeld et al. 2023; Calame et al. 2023; Nashef et al. 2026 (rate tracks velocity and deceleration; covariance tracks gain) | Confirms with a different metric family; not a selling point | High |
| **B10** The Ising advantage during movement comes from reach-locked rate modulation | **Known in method; new in system** | Tyrcha et al. 2013 (retina: time-varying-field independent model beats stationary coupled model); Granot-Atedgi et al. 2013; Okun et al. 2012; Schwab et al. 2014 | Shown for a motor population with held-out likelihood per epoch | High |
| **Late post-reach coupling epoch** (2022 cohort only, suggestive) | **Not found in prior work; partly echoed** | Nashef et al. 2023 report decreaser-pair synchrony with an uptick before endpoint that persists after it; Cao et al. 2017 (slow ~1-s correlated rate fluctuations in PCs) | If replicated, a slow post-movement coupling epoch would be new. The lag-200 ms control already points to slow shared fluctuations of the Cao et al. kind | Low (suggestive result; prior work only partly matches) |

---

## 2. Per-claim discussion

### P1. Coupling beyond shared rate, present at rest

**What the literature says.** Simple-spike correlations between nearby PCs are long established: on-beam pairs are time-locked in reaching rats (Heck et al. 2007), ~200-Hz population oscillations synchronize PCs (de Solages et al. 2008), simple-spike synchrony has distinct mechanisms from complex-spike synchrony (Wise et al. 2010), and awake mouse PC pairs show strong slow (~1 s) rate co-fluctuations (Cao et al. 2017). In this very dataset, Nashef et al. (2023, 2026) find above-shuffle 1-ms co-firing in a large share of pairs. Herzfeld et al. (2023) find real but small millisecond synchrony in macaque flocculus. On the method side, separating stimulus-driven (rate) correlation from "intrinsic" coupling with time-dependent fields is an established retinal programme (Granot-Atedgi et al. 2013; Ferrari et al. 2018; Delamare & Ferrari 2022). Tyrcha et al. (2013) showed that ignoring nonstationary drive inflates inferred couplings.

**How we differ.** (i) Model class: no pairwise max-ent/Ising fit to PCs was found, so we may be the first to apply the Schneidman/Tkačik framework to cerebellar cortex. (ii) Control: the coupling advantage is measured against a time-varying-rate independent model with held-out likelihood (A7: 21/21 sessions). This is stronger than the shift-predictor or shuffle corrections in the synchrony papers, which Herzfeld et al. (2023) argue confound rate and covariance. (iii) Resolution: 10-ms binarized states (same-bin co-variation), between the 1-ms synchrony and the ~1-s slow fluctuations of the earlier work.

**Reviewer risk.** High for "PCs are correlated beyond rate" stated plainly. That is known, including in this dataset. Low to moderate for the model-based form. Frame P1 as a population-model quantification, not a discovery of coupling. One caution: Nashef et al. restrict pairs to non-overlapping waveform channels. A reviewer may ask about duplicate or shared-channel units (the results log notes 220516 has many r > 0.5 pairs). A7 survives pair pruning, which answers this.

### P2. No coupling increase during reach at 10 ms; movement is carried by fields

**Artefact half (methodologically known).** That nonstationary rates leak into stationary-model couplings or correlograms has a long history: Brody (1999) on latency and excitability covariation, Ventura et al. (2005) on time-varying dependence, Tyrcha et al. (2013) for kinetic Ising, Okun et al. (2012) for population-rate-driven patterns, and Schwab et al. (2014) for latent drives. Ferrari et al. (2018) show that, once stimulus drive is modelled, the same retinal coupling network appears across very different stimulus ensembles. That is a direct precedent for "J fixed, fields carry the change". Our contribution is to show this in a motor population, with a concrete warning: a stationary Ising model on PCs would wrongly report coupling that rises during reach (B2 and its model-dependence note).

**Biological half (contested).** Several papers report synchrony increasing during movement:
- Nashef et al. (2023/2026): same mice and recordings, 1-ms co-firing, strongest near endpoint and in rate-suppressed cells.
- Sedaghat-Nejad et al. (2022): marmoset PC synchrony during saccade deceleration while rate falls.
- Heck et al. (2007): movement-locked on-beam synchrony.
- Brown et al. (2024): synchronous sensory-evoked gaps.
- Wagner et al. (2021): complex-spike (not simple-spike) synchrony emerging around reach onset.

Herzfeld et al. (2023) support our side: synchrony is small, a rate code dominates, and standard metrics conflate rate with covariance.

**Reviewer risk: high.** The reconciliation needs to be in the paper (Section 3). Note that A8 is not powered to exclude a Nashef-sized effect. The acceleration-epoch CI runs from −28% to +93%, and the fitted reach-epoch offset leans positive (+44%, 17/21), which points the same way as Nashef et al. Our wording ("no evidence… moderate changes not excluded") is correctly hedged. Do not write "contradicts Nashef".

### P3. Field-energy dip at peak acceleration without a mean-rate change

**What the literature says.** Opposite-sign PC responses whose sum cancels are central to recent cerebellar population work. Herzfeld et al. (2015) showed that bursting and pausing PCs combined, not either alone, predict saccade speed. Fakharian et al. (2025) show that individual PC "vectors" cancel at the population level. In this dataset, Calame et al. (2021/2023) found roughly equal bursters and pausers. When pooled, the broad individual responses cancelled except for a sharp net pause at peak outward acceleration, scaled by velocity. They read this as disinhibition preceding the nuclear deceleration burst (Becker & Person 2019).

**How we differ.** (i) Calame et al. pool all PCs across animals and bin reaches by velocity. Our signal is per session (20/21) and needs no pooling, velocity binning or burster/pauser labels. (ii) Our quantity is the field term of the population energy, which weights each neuron's state by its own field. So it is sensitive to the *pattern* of who is up and who is down, which a sum of rates is not. That is the defensible novelty.

**Risk: moderate-high, from two sides.** First, the timing is not new and Calame et al. must be cited as the first report of a population event at peak acceleration in these data. Second, the claim that "mean rate barely changes" at peak acceleration seems to conflict with Calame's pooled drop. Possible reasons: per-session versus pooled averaging, their velocity-binned display (the drop scales with velocity, so it is small for slow reaches), normalization, or cell inclusion criteria. Before submission, compute our population-mean rate in the same way as Calame Fig. 1i–j (pooled across sessions, velocity-binned). Then report either that the energy dip persists after regressing out the pooled-rate pause, or reword to "is larger and more consistent across sessions than the mean-rate change".

### P4. Polarization lowers coupling influence at fixed J

**What the literature says.** The ingredients are known:
- For binary units, covariance scales with p(1−p), and measured correlation depends on rate even at fixed input correlation (de la Rocha et al. 2007; Cohen & Kohn 2011).
- In Ising models, strong fields saturate units and suppress susceptibility (textbook).
- Inputs quench variability across cortex (Churchland et al. 2010).
- Collective/critical signatures of fitted max-ent models depend on rates and shared drive (Tkačik et al. 2015; Mora & Bialek 2011; Nonnenmacher et al. 2017; Schwab et al. 2014; Macke et al. 2011).
- Time-resolved max-ent macroscopic quantities (entropy, heat capacity, sparseness, how much pairwise interactions add) have been tracked during stimulus processing (Donner et al. 2017; Gaudreault & Shimazaki 2018).

**What appears new.** I found no report that a motor population changes its operating point *during movement*, becoming field-dominated so that fixed couplings contribute less multi-information or susceptibility, nor any such report for PCs. Combined with B10 and P2, it gives a coherent story: coupling is a stable substrate, and movement pushes the population into a regime where rates dominate.

**Risk: moderate.** Reviewers may call it tautological: with J fixed and |h| increasing, susceptibility must fall. To answer this:
- show that the polarization itself is an empirical fact (p(1−p) falls in 17/21), not something the model imposes;
- report the size of the drop in coupling influence relative to a surrogate in which each neuron's rate change is shuffled across time, preserving the mean p(1−p) trajectory;
- cite Nonnenmacher et al. (2017) and Schwab et al. (2014) to show you know that rate-driven signatures can mimic collective ones.

### P5. Mossy-fiber perturbation: coupling or only rates?

**What the literature says.** Calame et al. (2023), on these recordings, report rate-level effects only: stimulation drives increases in some PCs and decreases in others, the effects fade over the block, and aftereffects appear in washout. Guo et al. (2021) stimulated pontine nuclei during reach and found cerebellar and cortical activity "largely preserved" (rate-level). Direct PC optogenetics (Witter et al. 2013; Heiney et al. 2014; Hoogland et al. 2015) imposes synchrony rather than measuring it. Wagner et al. (2021) changed complex-spike synchrony by perturbing cerebellar feedback to the olive. That is a different input and spike type, but it shows a perturbation can shift a synchrony state. Outside cerebellum, Ising couplings change with experience in about 20% of PFC sessions (Tavoni et al. 2017), while retinal couplings stay invariant across stimuli (Ferrari et al. 2018).

**Verdict.** No in-vivo study found that measures PC simple-spike coupling under mossy-fiber or pontine optogenetics or across an adaptation block. Either outcome is reportable:
- If κ changes, it is the first evidence that a learned mossy-fiber perturbation reorganizes PC coupling.
- If κ is unchanged, it fits the "stable coupling substrate, rate-driven adaptation" view (Ferrari et al. 2018) and extends P2 to perturbation.

The preliminary per-block look (Results log, section C) is not FDR-significant. Rate changes during stimulation are large and divergent, so the P2 artefact is a real threat: only the joint time-varying-field model (stimfit) should be used.

### B6, B10 and the late epoch

B6 matches the rate-code consensus (Herzfeld et al. 2023) and Nashef et al. (2026), where rate tracks velocity and deceleration. It supports our framing but is not new. B10 repeats in PCs what Tyrcha et al. (2013) found in retina: a time-varying-field independent model explains much of what a stationary coupled model attributes to coupling. Cite as confirmation in a new system. On the late epoch, Nashef et al. (2023) report decreaser-pair synchrony that persists after endpoint, and Cao et al. (2017) report ~1-s correlated slow rate fluctuations in awake mouse PCs. Our lag-200-ms control rising in 6/6 sessions fits the slow-fluctuation account. Keep this result in the supplement as suggestive.

---

## 3. Literature that contradicts or complicates our claims

1. **Nashef et al. 2026, Nat Commun (and its 2023 preprint). Same dataset; PC co-firing increases during reach.** Directly opposes a simple reading of P2. Points to reconcile in the text:
   - **Timescale.** Their effect is mostly 1-ms synchrony: over half of co-firing is lost with 1-ms jitter, though some survives 1–75 ms jitter. Our 10-ms binarization mixes 1-ms synchrony with the remaining 9 ms, so a small exact-synchrony increase can be diluted.
   - **Selectivity.** The increase is concentrated in rate-suppressed and early-facilitated clusters and near endpoint (~60–100 ms before). Ours is a whole-population scalar averaged over all pairs and epochs.
   - **Rate control.** Their null is trial-order shuffle or shift predictor. Ours is a time-varying-field model, which also absorbs trial-to-trial co-fluctuation of rates that shuffles do not. Herzfeld et al. (2023) argue such metrics confound rate and covariance. Nashef et al. (2026) themselves concede that rate co-fluctuations partly drive their jPETH differences.
   - **Power.** Our CI does not exclude an increase of their size.

   Suggested sentence: "Our 10-ms population model detects no robust change in coupling during reach. That is compatible with the millisecond-precise synchrony increases reported in subsets of PC pairs in these recordings (Nashef et al. 2026), which our binning and population-level estimate are not designed to resolve."
2. **Calame et al. 2021/2023. Pooled PC rate drops sharply at peak acceleration.** This complicates P3's "mean rate barely changes" (Section 2, P3).
3. **Sedaghat-Nejad et al. 2022 (marmoset, saccades). Synchrony rises during deceleration while rate falls.** A second species and effector showing movement-phase synchrony. Same answer as for Nashef et al. (1-ms precision, subsets of pairs).
4. **Heck et al. 2007; Brown et al. 2024; Hoogland et al. 2015; Wagner et al. 2021.** Movement-, sensory- or learning-locked synchrony. Wagner et al. and Hoogland et al. concern complex spikes. State that our analysis treats binarized spiking that is dominated by simple spikes (confirm whether complex spikes were included in the binarized trains).
5. **Okun et al. 2012; Schwab et al. 2014; Nonnenmacher et al. 2017; Roudi et al. 2009.** Caution for the paper as a whole: population-pattern statistics and max-ent "collective" signatures can follow from population-rate dynamics or latent drive alone. Roudi et al. (2009) also show that pairwise models describe small populations well almost by default when N is below about 1/(rate × bin). With N = 3–32 and 10-ms bins, some sessions may sit in that regime. Report N × rate × bin per session.
6. **Ferrari et al. 2018.** Supports P2 and P4. A reviewer could also use it to argue that "J fixed while fields change" is expected, not surprising.

---

## 4. Gaps and suggested novelty framing

**Claims nobody appears to have made (within my coverage):**
1. A pairwise max-ent (Ising) description of cerebellar PC populations, with held-out comparison against a time-varying-rate independent model.
2. A demonstration in a motor structure that stationary max-ent models turn movement-locked rate changes into apparent coupling increases, removed by time-varying fields. This is a practical warning for anyone extending synchrony or coupling claims to movement.
3. A per-session population-pattern (field-energy) signature of peak acceleration, robust across 20/21 sessions, that a simple population-mean rate does not capture. This needs the Calame reconciliation first.
4. Movement as an operating-point shift (polarization) that lowers the influence of fixed couplings (P4), if it survives a surrogate control.
5. Coupling under closed-loop mossy-fiber perturbation and adaptation (P5), whatever the sign.

**Suggested novelty statement (draft):**
> Purkinje-cell synchrony during movement has so far been studied with pairwise spike-timing measures, with conflicting conclusions about whether it changes with behavior (Heck et al. 2007; Sedaghat-Nejad et al. 2022; Herzfeld et al. 2023; Nashef et al. 2026). Here we model PC populations with pairwise maximum-entropy models whose single-neuron fields vary over the reach. This separates shared rate drive from statistical coupling. PC populations show coupling beyond shared reach-locked rate modulation that is present at rest and, at 10-ms resolution, does not detectably strengthen during reaching. A stationary model would wrongly report such strengthening. Instead, movement acts on the fields: at peak acceleration PCs polarize and the population pattern shifts, which lowers the influence of the stable coupling network. This is a population-level signature that the mean firing rate does not capture.

**Phrases to avoid:** "first to show PCs are coupled"; "contradicts Nashef et al."; "mean rate is unchanged" (until the Calame comparison is done); "criticality".

**Analyses that would strengthen the novelty case:**
- Reproduce Calame Fig. 1i–j on our sessions and show the field-energy dip after partialling out the pooled-rate pause.
- Re-bin at 1–2 ms for the cluster-1-like (reach-suppressed) PCs and check whether the time-varying-field model recovers a Nashef-type increase near endpoint. A positive result would turn the main conflict into a scale-dependence finding.
- For P4, a surrogate that keeps each neuron's p(t) but randomizes which neurons polarize.
- Report N × rate × bin (Roudi et al. 2009) and the model's specific heat only with a Nonnenmacher-type subsampling control, if heat capacity is shown at all.

---

## 5. Bibliography (checked in this session; "DOI not seen" means the record was found but the DOI was not displayed)

### Cerebellar physiology: synchrony, correlation, population coding
1. Heck DH, Thach WT, Keating JG (2007). On-beam synchrony in the cerebellum as the mechanism for the timing and coordination of movement. *PNAS* 104:7658–7663. https://doi.org/10.1073/pnas.0609966104
2. de Solages C, et al. (2008). High-frequency organization and synchrony of activity in the Purkinje cell layer of the cerebellum. *Neuron* 58:775–788. https://doi.org/10.1016/j.neuron.2008.05.008
3. Wise AK, Cerminara NL, Marple-Horvat DE, Apps R (2010). Mechanisms of synchronous activity in cerebellar Purkinje cells. *J Physiol* 588:2373–2390. https://pmc.ncbi.nlm.nih.gov/articles/PMC2915514 (DOI not seen)
4. Person AL, Raman IM (2012). Purkinje neuron synchrony elicits time-locked spiking in the cerebellar nuclei. *Nature* 481:502–505. https://doi.org/10.1038/nature10732
5. Person AL, Raman IM (2012). Synchrony and neural coding in cerebellar circuits. *Front Neural Circuits* 6:97. https://doi.org/10.3389/fncir.2012.00097
6. Cao Y, et al. (2017). Cerebellar Purkinje cells generate highly correlated spontaneous slow-rate fluctuations. *Front Neural Circuits* 11:67. https://doi.org/10.3389/fncir.2017.00067
7. Lang EJ, Sugihara I, Welsh JP, Llinás R (1999). Patterns of spontaneous Purkinje cell complex spike activity in the awake rat. *J Neurosci* 19:2728–2739. https://doi.org/10.1523/JNEUROSCI.19-07-02728.1999
8. Herzfeld DJ, Kojima Y, Soetedjo R, Shadmehr R (2015). Encoding of action by the Purkinje cells of the cerebellum. *Nature* 526:439–442. https://doi.org/10.1038/nature15693
9. Herzfeld DJ, Kojima Y, Soetedjo R, Shadmehr R (2018). Encoding of error and learning to correct that error by the Purkinje cells of the cerebellum. *Nat Neurosci* 21:736–743. https://doi.org/10.1038/s41593-018-0136-y
10. Herzfeld DJ, Joshua M, Lisberger SG (2023). Rate versus synchrony codes for cerebellar control of motor behavior. *Neuron* 111:2448–2460. https://doi.org/10.1016/j.neuron.2023.07.002
11. Sedaghat-Nejad E, Pi JS, Hage P, Fakharian MA, Shadmehr R (2022). Synchronous spiking of cerebellar Purkinje cells during control of movements. *PNAS* 119:e2118954119. https://doi.org/10.1073/pnas.2118954119
12. Fakharian MA, Shoup, Hage P, Elseweifi, Shadmehr R (2025). A vector calculus for neural computation in the cerebellum. *Science* 388:869–875. https://doi.org/10.1126/science.adu6331
13. Hong S, Negrello M, Junker M, Smilgin A, Thier P, De Schutter E (2016). Multiplexed coding by cerebellar Purkinje neurons. *eLife* 5:e13810. https://doi.org/10.7554/eLife.13810
14. Brown ST, Raman IM (2018). Sensorimotor integration and amplification of reflexive whisking by well-timed spiking in the cerebellar corticonuclear circuit. *Neuron* 99:564–575. https://doi.org/10.1016/j.neuron.2018.06.028
15. Brown ST, Medina M, Holla M, Vaaga C, Raman IM (2024). Simple spike patterns and synaptic mechanisms encoding sensory and motor signals in Purkinje cells and the cerebellar nuclei. *Neuron*. https://www.cell.com/neuron/fulltext/S0896-6273(24)00124-7 (DOI and volume not seen; author list from the Northwestern news item)
16. Sauerbrei BA, Lubenov EV, Siapas AG (2015). Structured variability in Purkinje cell activity during locomotion. *Neuron* 87:840–852. https://doi.org/10.1016/j.neuron.2015.08.003
17. Gaffield MA, Christie JM (2017). Movement rate is encoded and influenced by widespread, coherent activity of cerebellar molecular layer interneurons. *J Neurosci* 37:4751–4765. https://doi.org/10.1523/JNEUROSCI.0534-17.2017
18. Wagner MJ, Savall J, Hernandez O, et al., Luo L, Ganguli S, Schnitzer MJ (2021). A neural circuit state change underlying skilled movements. *Cell* 184:3731–3747. https://doi.org/10.1016/j.cell.2021.06.001
19. Kostadinov D, Beau M, Blanco-Pozo M, Häusser M (2019). Predictive and reactive reward signals conveyed by climbing fiber inputs to cerebellar Purkinje cells. *Nat Neurosci* 22:950–962. https://doi.org/10.1038/s41593-019-0381-8
20. De Zeeuw CI, Romano V (2022). Time and tide of cerebellar synchrony (commentary). *PNAS*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9170046 (review; used only as a pointer)
21. Herzfeld DJ, Lisberger SG (2025). Neural circuit mechanisms to transform cerebellar population dynamics for motor control in monkeys. *bioRxiv*. https://doi.org/10.1101/2025.02.21.639459 (abstract only)
22. Agarwal et al. (2024). Highly synchronized inhibition from Purkinje cells entrains cerebellar output in zebrafish. *bioRxiv*. https://doi.org/10.1101/2024.06.27.600928

### Reach task, this dataset, and perturbation
23. Calame DJ, Becker MI, Person AL (2023). Cerebellar associative learning underlies skilled reach adaptation. *Nat Neurosci* 26:1068–1079. https://doi.org/10.1038/s41593-023-01347-y (preprint: https://doi.org/10.1101/2021.12.17.473247)
24. Nashef A, Spindle MS, Calame DJ, Person AL (2023). A dual Purkinje cell rate and synchrony code sculpts reach kinematics. *bioRxiv*. https://doi.org/10.1101/2023.07.12.548720 (PMC10370034)
25. Nashef A, Spindle MS, Calame DJ, Person AL (2026). Behavioral correlates of Purkinje cell ensemble covariance. *Nat Commun* 17:7755. https://doi.org/10.1038/s41467-026-74576-x (DOI confirmed via Crossref)
26. Becker MI, Person AL (2019). Cerebellar control of reach kinematics for endpoint precision. *Neuron* 103:335–348. https://doi.org/10.1016/j.neuron.2019.05.007
27. Becker MI, Calame DJ, Wrobel J, Person AL (2020). Online control of reach accuracy in mice. *J Neurophysiol* 124:1637–1655. https://pmc.ncbi.nlm.nih.gov/articles/PMC7814908 (DOI not seen)
28. Guo JZ, Sauerbrei BA, Cohen JD, Mischiati M, Graves AR, Pisanello F, Branson KM, Hantman AW (2021). Disrupting cortico-cerebellar communication impairs dexterity. *eLife* 10:e65906. https://doi.org/10.7554/eLife.65906
29. Witter L, Canto CB, Hoogland TM, de Gruijl JR, De Zeeuw CI (2013). Strength and timing of motor responses mediated by rebound firing in the cerebellar nuclei after Purkinje cell activation. *Front Neural Circuits* 7:133. https://doi.org/10.3389/fncir.2013.00133
30. Heiney SA, Kim J, Augustine GJ, Medina JF (2014). Precise control of movement kinematics by optogenetic inhibition of Purkinje cell activity. *J Neurosci* 34:2321–2330. https://doi.org/10.1523/JNEUROSCI.4547-13.2014
31. Hoogland TM, de Gruijl JR, Witter L, Canto CB, De Zeeuw CI (2015). Role of synchronous activation of cerebellar Purkinje cell ensembles in multi-joint movement control. *Curr Biol* 25:1157–1165. https://doi.org/10.1016/j.cub.2015.03.009
32. Chabrol FP, Blot A, Mrsic-Flogel TD (2019). Cerebellar contribution to preparatory activity in motor neocortex. *Neuron* 103:506–519. https://doi.org/10.1016/j.neuron.2019.05.022

### Maximum-entropy, nonstationarity, time-varying couplings
33. Schneidman E, Still S, Berry MJ, Bialek W (2003). Network information and connected correlations. *Phys Rev Lett* 91:238701. https://doi.org/10.1103/PhysRevLett.91.238701
34. Schneidman E, Berry MJ, Segev R, Bialek W (2006). *Nature* 440:1007–1012. https://doi.org/10.1038/nature04701 (in the internal review)
35. Tkačik G, Marre O, Amodei D, Schneidman E, Bialek W, Berry MJ (2014). *PLoS Comput Biol* 10:e1003408. https://doi.org/10.1371/journal.pcbi.1003408 (in the internal review)
36. Granot-Atedgi E, Tkačik G, Segev R, Schneidman E (2013). Stimulus-dependent maximum entropy models of neural population codes. *PLoS Comput Biol* 9:e1002922. https://doi.org/10.1371/journal.pcbi.1002922
37. Tyrcha J, Roudi Y, Marsili M, Hertz J (2013). The effect of nonstationarity on models inferred from neural data. *J Stat Mech* P03005. https://arxiv.org/abs/1203.5673 (DOI not seen)
38. Ferrari U, Deny S, Chalk M, Tkačik G, Marre O, Mora T (2018). Separating intrinsic interactions from extrinsic correlations in a network of sensory neurons. *Phys Rev E* 98:042410. https://doi.org/10.1103/PhysRevE.98.042410
39. Delamare G, Ferrari U (2022). Time-dependent maximum entropy model for populations of retinal ganglion cells. *arXiv*:2207.12810. https://arxiv.org/abs/2207.12810
40. Shimazaki H, Amari S, Brown EN, Grün S (2012). State-space analysis of time-varying higher-order spike correlation for multiple neural spike train data. *PLoS Comput Biol* 8:e1002385. https://doi.org/10.1371/journal.pcbi.1002385
41. Donner C, Obermayer K, Shimazaki H (2017). Approximate inference for time-varying interactions and macroscopic dynamics of neural populations. *PLoS Comput Biol* 13:e1005309. https://doi.org/10.1371/journal.pcbi.1005309
42. Gaudreault J, Shimazaki H (2018). State-space analysis of an Ising model reveals contributions of pairwise interactions to sparseness, fluctuation, and stimulus coding of monkey V1 neurons. *ICANN 2018*, LNCS. https://doi.org/10.1007/978-3-030-01424-7_63
43. Gardella C, Marre O, Mora T (2016). A tractable method for describing complex couplings between neurons and population rate. *eNeuro* 3:ENEURO.0160-15.2016. https://www.eneuro.org/content/3/4/ENEURO.0160-15.2016
44. Meshulam L, Gauthier JL, Brody CD, Tank DW, Bialek W (2017). Collective behavior of place and non-place neurons in the hippocampal network. *Neuron* 96:1178–1191. https://doi.org/10.1016/j.neuron.2017.10.027
45. Tavoni G, Ferrari U, Battaglia FP, Cocco S, Monasson R (2017). Functional coupling networks inferred from prefrontal cortex activity show experience-related effective plasticity. *Netw Neurosci* 1:275–301. https://doi.org/10.1162/NETN_a_00014
46. Roudi Y, Nirenberg S, Latham PE (2009). Pairwise maximum entropy models for studying large biological systems: when they can work and when they can't. *PLoS Comput Biol* 5:e1000380. https://doi.org/10.1371/journal.pcbi.1000380
47. Truccolo W, Hochberg LR, Donoghue JP (2010). Collective dynamics in human and monkey sensorimotor cortex: predicting single neuron spikes. *Nat Neurosci* 13:105–111. https://www.nature.com/articles/nn.2455
48. Stevenson IH, et al., Kording KP (2012). Functional connectivity and tuning curves in populations of simultaneously recorded neurons. *PLoS Comput Biol* 8:e1002775. https://doi.org/10.1371/journal.pcbi.1002775

### Operating point, criticality, rate dependence of correlation
49. Tkačik G, Mora T, Marre O, Amodei D, Palmer SE, Berry MJ, Bialek W (2015). Thermodynamics and signatures of criticality in a network of neurons. *PNAS* 112:11508–11513. https://doi.org/10.1073/pnas.1514188112
50. Mora T, Bialek W (2011). Are biological systems poised at criticality? *J Stat Phys* 144:268–302. https://doi.org/10.1007/s10955-011-0229-4
51. Hidalgo J, Grilli J, Suweis S, Muñoz MA, Banavar JR, Maritan A (2014). Information-based fitness and the emergence of criticality in living systems. *PNAS* 111:10095–10100. https://doi.org/10.1073/pnas.1319166111
52. Schwab DJ, Nemenman I, Mehta P (2014). Zipf's law and criticality in multivariate data without fine-tuning. *Phys Rev Lett* 113:068102. https://arxiv.org/abs/1310.0448
53. Nonnenmacher M, Behrens C, Berens P, Bethge M, Macke JH (2017). Signatures of criticality arise from random subsampling in simple population models. *PLoS Comput Biol* 13:e1005718. https://doi.org/10.1371/journal.pcbi.1005718
54. Macke JH, Opper M, Bethge M (2011). Common input explains higher-order correlations and entropy in a simple model of neural population activity. *Phys Rev Lett* 106:208102. https://arxiv.org/abs/1009.2855
55. de la Rocha J, Doiron B, Shea-Brown E, Josić K, Reyes A (2007). Correlation between neural spike trains increases with firing rate. *Nature* 448:802–806. https://doi.org/10.1038/nature06028
56. Cohen MR, Kohn A (2011). Measuring and interpreting neuronal correlations. *Nat Neurosci* 14:811–819. https://doi.org/10.1038/nn.2842
57. Doiron B, Litwin-Kumar A, Rosenbaum R, Ocker GK, Josić K (2016). The mechanics of state-dependent neural correlations. *Nat Neurosci* 19:383–393. https://doi.org/10.1038/nn.4242
58. Churchland MM, et al. (2010). Stimulus onset quenches neural variability: a widespread cortical phenomenon. *Nat Neurosci* 13:369–378. https://doi.org/10.1038/nn.2501

### Rate–covariance confounds and synchrony versus rate in motor cortex
59. Brody CD (1999). Correlations without synchrony. *Neural Comput* 11:1537–1551. https://doi.org/10.1162/089976699300016133
60. Ventura V, Cai C, Kass RE (2005). Trial-to-trial variability and its effect on time-varying dependency between two neurons. *J Neurophysiol* 94:2928–2939. https://doi.org/10.1152/jn.00644.2004
61. Okun M, Yger P, Marguet SL, et al., Carandini M, Harris KD (2012). Population rate dynamics and multineuron firing patterns in sensory cortex. *J Neurosci* 32:17108–17119. https://doi.org/10.1523/JNEUROSCI.1831-12.2012
62. Riehle A, Grün S, Diesmann M, Aertsen A (1997). Spike synchronization and rate modulation differentially involved in motor cortical function. *Science* 278:1950–1953. https://doi.org/10.1126/science.278.5345.1950

**Unverified or not found (not relied on):** Bell & Grimm 1969 (seen only in a secondary citation); "Payne et al. 2019" (not found); Lee et al., Tsubota et al., Proville et al. 2014, Medina & Lisberger, Han et al. 2020 (not searched).
