# What could be novel about Purkinje-cell populations from a statistical-mechanics model (2026-10-07)

## Starting point: what is not novel on its own
- "Neurons are correlated beyond independent firing" and "a pairwise Ising model beats an independent model" have been shown many times (retina, cortex). For Purkinje cells (PCs), simple-spike synchrony is also an established topic: Person & Raman 2011 (10.1038/nature10732), Ramirez & Stell 2016 (10.1016/j.celrep.2016.11.075), Nashef et al. 2023 (10.1101/2023.07.12.548720), Sedaghat-Nejad et al. 2021 (10.1101/2021.09.16.460700).
- Time-varying fields in maximum-entropy models also have precedent: Granot-Atedgi et al. 2013, stimulus-dependent maxent (10.1371/journal.pcbi.1002922), and Shimazaki et al. 2012, state-space Ising with time-varying interactions (10.1371/journal.pcbi.1002385).
- A quick OpenAlex search (8 queries) found **no pairwise maximum-entropy/Ising model fitted to PC spiking populations**. That is suggestive, not proof; a full search is needed before claiming "first".

So the novelty has to be a PC-specific biological result that requires a model of the joint population pattern. A pairwise correlation, GLM or latent-factor model cannot give it.

## What only a joint-pattern (stat-mech) model gives
1. **A probability for every population pattern**, e.g. "K of N cells silent together in this 10-ms bin", conditioned on the time-varying rates.
2. **Counterfactuals:** switch the couplings off (J = 0) with rates unchanged and ask how pattern probabilities change. This measures what coupling adds to the population output.
3. **An additive split of each pattern's log-probability** into rate (field) and coupling parts, tested on held-out reaches against a shuffle null.

## Candidate novel claims, ranked

| # | Claim to test | Why it is PC-specific and needs the model | Data we have | Risk |
|---|---|---|---|---|
| 1 | **PC coupling is a co-silence (synchronous-pause) structure.** Beyond time-varying rates, coupling mainly makes cells *pause together*, not fire together. | PCs fire tonically (baseline spike probability about 0.60 per 10-ms bin), so pauses are the signal. Cerebellar-nuclei neurons are driven by synchronous PC pauses (Person & Raman 2011). The model gives P(K cells silent) with vs without J at matched rates. That is the input statistic the nuclei see, which pairwise correlations cannot give. | Yes. The per-bin co-active/co-silent/discordant terms already exist (j_coact, j_cosil, j_disc), but only for the stationary model | Low–medium |
| 2 | **Coupling is unchanged when mossy-fiber input is perturbed and during adaptation.** | The dataset has an optogenetic perturbation of the main excitatory input. No published Ising study has a causal input manipulation. If J is unchanged, coupling arises from local or olivary circuitry (MLIs, climbing fibers), not mossy-fiber drive. If J rises in the pulse window, shared mossy-fiber input generates it. Either outcome is informative. | Needs the stimulation timing (in progress) and the joint model with a coupling scale factor κ (validated on synthetic data) | Medium; power may limit it |
| 3 | **Rate carries movement; coupling is a stable scaffold.** | This addresses the "dual rate + synchrony code" (Nashef 2023): at 10 ms, synchrony beyond rate is not modulated by the reach. The conflict is only interesting if tested at their timescale. | Yes at 10 ms. Needs a bin-size sweep (2, 5, 10, 20 ms) with time-varying fields | Medium; 2-ms bins are sparse |
| 4 | **Coupling reflects climbing-fiber/complex-spike structure.** | Complex spikes are synchronized across a microzone (olivary gap junctions) and are followed by a 10–20 ms simple-spike pause. Synchronized complex spikes would create co-silence coupling at 10 ms. That would give a mechanism for claim 1 and be a new link between complex-spike synchrony and simple-spike population structure. | **Unknown**: are complex spikes sorted, removed, or included in the spike trains? | Could also be a confound |
| 5 | **Coupling maps functional microzones.** J falls off with distance, and sign/cluster structure aligns with parasagittal zones. | The model's couplings are conditional (they control for all other neurons), unlike raw correlations. | Needs unit positions on the probe | Medium |
| 6 | **Pairwise is sufficient for PCs.** With time-varying fields, a pairwise model predicts held-out triplets and P(K). | Says the PC population output has no higher-order synergy at this scale | Yes (triplet files exist for the stationary model) | Incremental |
| 7 | Energy landscape / criticality (heat-capacity peak, ground state "all firing") | Thermodynamic signature | Yes | High: known artefacts from latent shared inputs; the pilot did not replicate |

### Added 2026-10-07: time-resolved thermodynamics of the reach (candidate 8)
- The energy dip and recovery over the reach (time-varying-field model) is mostly a restatement of the rate change: h(t) is fitted to follow the rates, so −Σh_i(t)s_i must move when the rates move. On its own it is not new evidence.
- What would be new is the model's own thermodynamic quantities at each time t, computed from h(t) and the fixed J:
  (a) multi-information I(t) = S_independent(t) − S_Ising(t), i.e. how much the couplings constrain the population at that moment;
  (b) susceptibility / heat capacity at each t, i.e. how strongly the population responds to a small shared input at that moment;
  (c) model surprise, E_observed(t) − ⟨E⟩_model(t): is the population more or less typical than the model expects, which would reveal structure the model misses (higher-order or time-varying J).
- Hypothesis: the reach-related pause moves PCs from their tonic baseline (median spike probability about 0.60 per 10-ms bin) toward about 0.43, through p ≈ 0.5, where single-neuron variance is maximal. Fixed couplings would then have their largest effect on joint patterns during the reach, even though J does not change. That would be "coupling's influence changes because rates move the population into a more susceptible regime". Only a stat-mech model can state and test this.
- Feasible now: exact for N ≤ 20 (39 conditions), Monte Carlo above that.

## Things we may have overlooked
- **The ±1 coding treats a pause and a spike symmetrically.** For tonically active PCs, the biologically meaningful event is the pause. We have never asked which half of the coupling (co-silence vs co-firing) survives the time-varying-field control.
- **Complex spikes** (claim 4) could be the source of much of the coupling. They need to be checked before any mechanism is claimed.
- **The 2022 cohort** (2 mice, much higher correlations) could be biology (a different lobule or microzone) or recording (sorting, duplicates). It must be resolved, and statistics should treat mouse as the unit (6 mice).
- **The stimulation dataset is a perturbation experiment.** That is rarer and stronger than another observational fit.

## Suggested first analyses (no new data needed for the first two)
1. Co-silence vs co-firing under the time-varying-field model. Held-out P(K silent) for M1 vs M0, the coupling counterfactual, by phase and cohort.
2. Bin-size sweep (2/5/10/20 ms) of the held-out coupling gain, overall and during the reach.
3. Once available: complex-spike times (claim 4), stimulation timing (claim 2), unit positions (claim 5).

## Possible one-paper framing (if 1 and 2 hold)
"Purkinje-cell populations in behaving mice carry a stable, pause-dominated coupling structure, beyond their shared rate modulation, that is invariant to perturbation of mossy-fiber input and to motor adaptation, while reach kinematics are carried by firing-rate fields."
