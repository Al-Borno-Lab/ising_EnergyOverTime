# Methods comparison: how our time-varying pairwise maximum-entropy model compares with other ways of studying coordinated Purkinje-cell activity

> Agent-generated 2026-10-12 (background research agent). Citations were checked by the agent against Crossref/PubMed/publisher records. **Open discrepancy:** the novelty report cites Nashef, Spindle, Calame & Person 2026, *Nat Commun* 17:7755 (DOI 10.1038/s41467-026-74576-x) as the published version of the 2023 preprint; the methods report and a follow-up search found only the bioRxiv preprint (PMC10370034). Verify before citing.

*Prepared 2026-10-12 for the Purkinje-cell (PC) Ising paper. Builds on `09_claude_science_handoff/reports/literature_review_coupling_methods.md` (method landscape and the field/coupling decomposition) and does not repeat it. Result IDs (A7, B10, etc.) refer to `RESULTS_LOG_FOR_PAPER.md`. Results listed in HANDOFF §7–8 but not yet in that log are cited as "HANDOFF §7.x". Every reference in §6 was checked against Crossref or the publisher page.*

---

## 0. The short version

- Most of what is known about PC coordination comes from **pair-level synchrony measures** at millisecond resolution: cross-correlograms, JPSTHs and synchrony indices (de Solages 2008; Heck 2007; Wise 2010; Sedaghat-Nejad 2022; Nashef 2023). These are model-free and fine-grained in time. Their weak point is that firing-rate changes that are shared across cells look like synchrony unless a shuffle or shift predictor removes them, and those corrections only remove the *trial-averaged* part.
- **GLMs with coupling terms** and **latent-variable models** (GPFA, LFADS, shared-gain models) are the main modern alternatives. A GLM with a time-varying baseline can answer roughly the same "is there coupling beyond reach-locked rates?" question that we answered. Our M1 pseudo-likelihood fit is, mechanically, a set of logistic GLMs with *same-bin* coupling. So we should not claim that *existence* of coupling beyond rates is unique to our method.
- What the thermodynamic approach adds is a **single, normalized, generative model of whole-population patterns in each 10-ms bin**. That gives us:
  1. exact held-out likelihoods for nested models (rate-only, static-coupling, and time-varying fields with fixed J);
  2. per-bin quantities in the same units (multi-information I(t), coupling susceptibility χ_J(t), heat capacity), which say how much the *fixed* couplings shape population patterns *right now*;
  3. an explicit **operating-point** account, in which coupling influence can change because firing probabilities move while J stays fixed;
  4. a one-parameter **κ test** of whether a perturbation changes coupling and not just rates.
- The main costs:
  - exact computation only up to N ≈ 20;
  - 10-ms binarization, which discards ms synchrony and spike counts;
  - pairwise, symmetric, same-bin coupling only;
  - J is *statistical* coupling, which most likely reflects shared input rather than synapses;
  - about 17 reaches per condition;
  - the pseudo-likelihood shortcut is unreliable in strongly coupled sessions.

---

## 1. Comparison table

Columns: **Q**, the question the method answers. **Sep.**, whether it separates shared rate co-modulation from coupling. **Time**, whether it gives time-resolved quantities. **Causal**, whether it supports a perturbation test. N/T/bin gives typical data needs.

| Method | Q | Key assumptions | N / trials / bin | Sep. | Time | Causal | Main pitfall | PC use |
|---|---|---|---|---|---|---|---|---|
| **Cross-correlogram (CCG), synchrony index (SI)** | Do two cells fire together more than chance, and at what lag? | Stationarity within the analysis window; chance level = product of rates | Pairs; tens of trials or long recordings; 0.5–2 ms bins | Only with a shift or shuffle predictor, and then only for the trial-averaged part | Coarse (per window) | Compare SI across conditions; confounded by rate | SI depends on rate; slow covariation produces broad CCG peaks (Brody 1999) | Main tool: de Solages 2008, Heck 2007, Wise 2010, Sedaghat-Nejad 2022, Nashef 2023 |
| **JPSTH** (Aertsen 1989) | How does pair correlation change over time relative to an event? | Repeated, aligned trials | Pairs; ≥ 50 trials preferred; 1–10 ms | Yes, the PSTH-product predictor removes trial-averaged rate | **Yes** (2-D in time) | Indirectly | Noisy with few trials; trial-to-trial excitability or latency changes look like synchrony (Ventura 2005) | Nashef 2023 (jPETH, 1-ms SI) |
| **Spike-time tiling coefficient (STTC)** (Cutts & Eglen 2014) | Rate-robust pairwise co-firing within ±Δt | Choice of Δt | Pairs; any length | Partly: robust to rate, not to rate *co-modulation* | No (per window) | No | Δt choice | Mainly retina and development |
| **Noise correlations** (r_sc) | Do trial-to-trial fluctuations co-vary? | Mean response fixed per condition | Pairs; many trials per condition; counting windows 10–1000 ms | Yes (removes the mean) | Per window | No | Mixes gain, state and coupling; depends on window and rate (de la Rocha 2007; Cohen & Kohn 2011) | Rarely used in PC work |
| **Shift / trial-shuffle predictor; unitary events** (Perkel 1967; Grün 2002) | Is there excess *precise* coincidence beyond what rates predict? | Null spike-train model (Poisson, dithering) | Pairs to ~10 cells; many trials; 1–5 ms | Yes for trial-averaged rate; dithering also for slow fluctuations (Louis 2010) | **Yes** (sliding windows) | Compare conditions | Choice of null; non-Poisson firing (PCs are very regular) inflates false positives | Trial shuffles used by Nashef 2023; we did not find a unitary-event study of PCs |
| **Coupled GLM** (Truccolo 2005; Pillow 2008) | Does neuron j's recent spiking predict neuron i's, beyond stimulus, history and baseline? | Conditional independence given the past; usually *lagged* coupling only | 10–100s of cells; minutes of data; 1–5 ms | **Yes** if the baseline is a time-varying (PSTH) term; trial-to-trial shared gain still loads into coupling | Usually no for coupling (fixed filters); time-varying versions exist | Can add perturbation terms | Unobserved common input appears as coupling (Kulkarni & Paninski 2007; Vidne 2012; Das & Fiete 2020) | Single-cell encoding GLMs/regressions (Chen 2016; Hong 2016); we did not find a coupled-GLM study of PC populations |
| **Stationary pairwise maxent / Ising** (Schneidman 2006; Shlens 2006) | How much of the population pattern distribution is explained by pairwise correlations? | Stationary distribution; pairwise constraints | N ≤ ~20 exact, more with MCMC; thousands of bins; 10–20 ms | **No.** Rate changes are absorbed into J (Tyrcha 2013; **seen in our data**, B2 note) | No | No | Non-stationarity; small-N extrapolation (Roudi 2009) | Our earlier pipeline; we did not find other PC studies |
| **K-pairwise maxent** (Tkačik 2014) | As above, plus matching the distribution of synchronous counts P(K) | Stationary | N up to ~100s (MCMC) | No | No | No | Same | None found |
| **Stimulus-dependent maxent** (Granot-Atedgi 2013) | Pairwise model whose fields follow a stimulus | Fields are a function of the stimulus; fixed J | Repeated stimulus; ~100s of trials | **Yes** (fields carry the stimulus) | Fields yes, J no | Possible | Needs many repeats; approximate fitting | None found. **Closest ancestor of our M1** |
| **State-space / kinetic Ising** (Shimazaki 2012; Roudi & Hertz 2011; Donner 2017) | How do (possibly directed, possibly higher-order) interactions change over time? | Markov dynamics (kinetic) or smooth parameter drift (state-space) | Many trials; N ≤ ~10–20 for exact state-space versions | Yes (time-varying fields) | **Yes, J(t) itself** | Possible | Many parameters; approximate inference; small-N | None found |
| **PCA, FA, GPFA, LFADS, dPCA** (Yu 2009; Pandarinath 2018; Kobak 2016) | What low-dimensional shared trajectories drive the population on single trials? | Linear-Gaussian (FA/GPFA) or RNN (LFADS) latent dynamics; smoothing | 10–1000s cells; 10s–100s trials; 10–50 ms smoothed | Partly. They capture shared fluctuations but do not separate fast pairwise coupling from shared drive | **Yes** | Latent-space perturbation analyses | Structure can follow from simple tuning (Elsayed & Cunningham 2017); spiking discreteness lost | Population decoding of PC SS (Herzfeld 2015, population grouped by complex-spike tuning) |
| **CCA / communication subspace** (Semedo 2019) | How do two populations share variability? | Linear | Two populations; many trials | Partly | Per window | No | Linear and Gaussian | Not in PCs as far as we found |
| **Shared-gain / latent-variable models** (Goris 2014; Ecker 2014; Lin 2015; Okun 2015) | How much covariation is one (or a few) shared multiplicative or additive gains? | Low-rank shared modulation | 10–100s cells; many trials | **Yes**: separates *trial-to-trial* shared gain from residual correlation | Usually per trial or epoch | No | A single gain cannot capture structured pairwise coupling | Not found in PCs |
| **Information theory: synergy/redundancy, PID** (Schneidman 2003; Williams & Beer 2010) | Is information about a variable shared, unique or synergistic across neurons? | Needs probability estimates; PID needs a choice of redundancy measure | Small N (2–3 sources); many trials | Not by itself | Per window | No | Bias with few samples; PID non-uniqueness | Not found in PCs |
| **Criticality / thermodynamic analogies** (Mora & Bialek 2011; Tkačik 2015) | Is the population near a critical point (heat-capacity peak at T ≈ 1)? | Stationary maxent fit plus an artificial temperature sweep | Large N; scaling with N | No | No | No | Latent common input produces critical signatures without fine-tuning (Schwab 2014; Aitchison 2016; Nonnenmacher 2017) | Our old pipeline's temperature sweep (criticality proxy did not replicate; RESULTS_LOG D) |
| **OURS: time-varying-field pairwise maxent, fixed J, exact ML + per-bin thermodynamics + κ** | Is there same-bin coupling beyond reach-locked rates? Does it change? How much does it shape population patterns at each moment? Does a perturbation change it? | Pairwise, symmetric, same-bin J; fields smooth in time (B-splines, 50-ms knots near the reach); bins conditionally independent given t; trial-averaged rates in h(t) | **N ≤ ~20 exact** (16-cell subsets above); 9–60 reaches (≈ 17 median); **10-ms binarized** | **Yes** for reach-locked rate (h(t)); trial-to-trial gain is checked only in simulation | **Yes** for fields and all thermodynamic quantities; J fixed by design (κ allows a scale change) | **Yes**, κ test with block-specific fields (E11; validated synthetically, real data running) | Binarization; small N; trials; shared input ≠ synapse; pseudo-likelihood fails moment matching in strongly coupled sessions | This project |

---

## 2. Method families

### 2.1 Pairwise correlation: CCGs, JPSTH, synchrony indices, STTC, noise correlations

**What it answers.** Whether two neurons fire within a few ms of each other more often than chance, at what lag (a CCG), and how that excess changes relative to an event (a JPSTH, Aertsen et al. 1989). The synchrony index (SI) used in PC work is the coincidence probability in a 1-ms bin divided by the product of the single-cell probabilities. Nashef et al. (2023) take it from Sedaghat-Nejad et al. (2022). STTC (Cutts & Eglen 2014) is a version designed to be insensitive to firing rate. Noise correlations measure whether trial-to-trial deviations from the mean response co-vary (Cohen & Kohn 2011).

**Assumptions.** Chance is defined by the single-cell rates in the same window or trial. For the JPSTH predictor, the trial-averaged PSTHs are assumed to describe each trial's rate.

**Data needs.** Pairs, so any N works, but the number of pairs grows as N². Millisecond bins are fine because each pair is analysed alone. Time-resolved versions (JPSTH, per-bin SI) need many trials: Nashef et al. tested every 1-ms bin with a Bonferroni-corrected paired test across many pairs.

**Rate versus coupling.** The PSTH-product (shift or shuffle) predictor removes *trial-averaged* rate co-modulation. It does not remove trial-to-trial co-variation in excitability or latency, which produces broad correlogram peaks (Brody 1999; Ventura, Cai & Kass 2005). SI also depends on firing rate, which Nashef et al. address with controls.

**Time resolution.** The JPSTH and per-bin SI are time-resolved. Correlograms and STTC are per window.

**Causal tests.** Only by comparing conditions, and a perturbation that changes rate also changes the expected coincidence count.

**Pitfalls.**
- Rate dependence: correlations rise with rate under fixed shared input (de la Rocha et al. 2007).
- Pairs say nothing about higher-order patterns: whether many PCs pause *together*, which matters for the nuclei (Person & Raman 2011).
- Correction for N² multiple comparisons.

**In cerebellum.** This is the dominant approach:
- millisecond synchrony and fast oscillations in the PC layer (de Solages et al. 2008);
- on-beam synchrony from shared parallel fibers (Heck et al. 2007);
- shared climbing-fiber and other input mechanisms (Wise et al. 2010);
- synchrony during saccade deceleration in marmosets (Sedaghat-Nejad et al. 2022);
- reach-related synchrony together with rate changes in mice, in the same lab and task family as our data (Nashef et al. 2023, bioRxiv);
- coordinated simple-spike pauses in calcium imaging (Ramirez & Stell 2016).

### 2.2 Shift / trial-shuffle predictors and unitary events

**What it answers.** Whether *precise* joint spike patterns (pairs or larger) occur more often than a null that keeps each neuron's rate profile (Perkel, Gerstein & Moore 1967; Gerstein & Perkel 1972; Grün, Diesmann & Aertsen 2002). The analysis uses sliding windows, so it is time-resolved.

**Assumptions.** A null process: Poisson with the local rate, or surrogates that dither spikes by a few ms (Louis et al. 2010).

**Data needs.** Many trials. Patterns over more than ~10 neurons become too rare to count.

**Rate versus coupling.** Trial shuffles remove trial-averaged rate. Dithering at ±Δ ms removes everything slower than Δ, so it separates *precise* synchrony from slow co-modulation. That is something our 10-ms model cannot do.

**Pitfalls.** PCs fire at high, very regular rates, so Poisson nulls are wrong. Dithering is needed, and its width sets what counts as "synchrony".

**In cerebellum.** Trial-shuffle predictors appear in Nashef et al. (2023). We did not find a unitary-event study of PCs.

**Relation to us.** Our trial-shuffle null (each neuron's reach order permuted) is the same idea as the shift predictor, applied to a population model. It is the null for A7 and for E10's H2.

### 2.3 GLMs with coupling filters

**What it answers.** Per neuron: P(spike_i at t | stimulus or kinematics, own history, other neurons' recent spikes) (Truccolo et al. 2005; Okatan, Wilson & Brown 2005; Pillow et al. 2008). The coupling filters are directed and lagged, and the held-out likelihood tests whether coupling adds prediction.

**Assumptions.** Neurons are conditionally independent given the past and the covariates. Standard GLMs therefore have **no same-bin coupling**: zero-lag synchrony must be explained by shared covariates or by lag-0 terms, which turns the model into a pseudo-likelihood Ising model.

**Data needs.** Many parameters per neuron. Scales to 100s of cells. 1–5 ms bins.

**Rate versus coupling.** Yes, if the baseline includes a time-varying reach-locked term, which is exactly what our M1 does with B-splines. Unobserved shared input still loads onto coupling filters (Kulkarni & Paninski 2007; Vidne et al. 2012), and inferred connectivity is systematically biased in strongly recurrent networks (Das & Fiete 2020; review: Stevenson et al. 2008).

**Time resolution.** Coupling filters are usually fixed in time. Time-varying coefficients are possible.

**Causal tests.** Perturbation covariates can be added, for example a laser regressor.

**In cerebellum.** GLMs and regressions are used mainly for single-cell encoding of kinematics (e.g. Chen et al. 2016 for whisker position; Hong et al. 2016 for multiplexed coding). We did not find a coupled-GLM analysis of simultaneously recorded PCs. **This is a gap a benchmark could fill (§5).**

**Important for our claims.** `tvfield_model.py`'s M1 is literally N logistic regressions (one per neuron) with spline fields and same-bin coupling to the other neurons' states. A GLM with a time-varying baseline and instantaneous coupling *is* our pseudo-likelihood model. A GLM with **lagged** coupling only is our Mlag control. With a 200-ms lag it captured only about a quarter of the gain (0.05 vs 0.18 nats/bin; RESULTS_LOG "How A7… fit together"). A one-bin-lag (10 ms) GLM has not been run, and it is the obvious benchmark.

### 2.4 Maximum-entropy / Ising models

**Stationary pairwise maxent** (Schneidman et al. 2006; Shlens et al. 2006).
- Fit the least-structured distribution over binary population words that matches every neuron's mean and every pair's correlation:

  P(s) ∝ exp(Σ h_i s_i + Σ_{i<j} J_ij s_i s_j).

- It answers how much of the structure in the pattern distribution pairs explain. Typically it uses N ≤ ~20 exact and 10–20-ms bins.
- **Main pitfall:** it assumes a stationary distribution. When all neurons change rate together over a trial, the pooled pairwise correlation rises and the model has to put that into J (Tyrcha et al. 2013).
- **We saw this directly.** In our stationary fits, the movement-related energy drop is carried by the coupling term (B2), and J looks like it rises across the reach (B3). With time-varying fields the drop moves into the field term: about −0.17 vs −0.01 per neuron at peak, field 20/21 sessions (B2 note). The stationary "J rises" is therefore an artefact (HANDOFF §7.2).

**K-pairwise** (Tkačik et al. 2014). Adds a term that matches P(K), the distribution of how many cells are active at once. It is needed at N ≈ 100. At our N the plain pairwise model already reproduces P(K) (Ising beats independent in 62/63 blocks; HANDOFF §7.1).

**Stimulus-dependent maxent** (Granot-Atedgi et al. 2013). The fields depend on the stimulus and J is fixed. This is the **direct conceptual ancestor of our M1**. The differences:
- our "stimulus" is time relative to the reach (spline fields), with no stimulus model;
- we fit by exact likelihood, not approximations;
- we add per-bin thermodynamic readouts and a perturbation test.

**Time-dependent maxent.**
- Shimazaki et al. (2012) fit a state-space model in which *both* fields and interactions, including higher-order ones, drift over trials or time.
- The kinetic (non-equilibrium) Ising model (Roudi & Hertz 2011) has directed, one-step-lagged couplings.
- Donner, Obermayer & Shimazaki (2017) give approximate inference for time-varying interactions in larger populations.
- These let J(t) change freely, which is more flexible than our fixed J. The cost is many more parameters and, at our trial counts (~17 per condition), poor power. Our H0 validation found that even per-pair epoch-specific J could not detect a true 1.6× coupling increase at N = 15 (prereg amendment 2). We therefore deliberately test **one** degree of freedom (κ) instead of a free J(t).

**Use in cerebellum.** We did not find another maxent analysis of PC populations, though the search was not exhaustive. Other reviews of the family: Gardella, Marre & Mora (2019); Roudi, Nirenberg & Latham (2009) on when small-N pairwise models can and cannot be trusted.

### 2.5 Dimensionality reduction, latent dynamics and decoding

**What they answer.** PCA, factor analysis, GPFA (Yu et al. 2009), LFADS (Pandarinath et al. 2018), dPCA (Kobak et al. 2016) and CCA / communication subspaces (Semedo et al. 2019) find a few shared latent signals that explain population covariance, often on single trials. Decoding approaches ask what the population encodes. For PCs, Herzfeld et al. (2015) grouped cells by complex-spike tuning and showed that the population simple-spike response predicts saccade kinematics.

**Assumptions.**
- Linear-Gaussian observations (FA, GPFA), or Poisson observations and RNN dynamics (LFADS).
- Smoothness.
- Low dimensionality.

**Data needs.** They scale to large N and need tens to hundreds of trials. LFADS needs much more data than ~17 reaches per condition.

**Rate versus coupling.** Partly. A latent that varies trial to trial captures shared fluctuations. These models have no notion of fast pairwise interaction, and a single-trial latent cannot tell shared drive from coupling. Population structure can also follow from simpler tuning (Elsayed & Cunningham 2017).

**Time resolution.** Yes, and on single trials.

**Causal tests.** Perturbations can be projected into the latent space, but there is no "coupling" parameter to test.

**Pitfalls.** Smoothing (GPFA typically uses 20–50-ms bins and smooth kernels) blurs exactly the same-bin co-firing that J measures. The discreteness of spiking (binary words, synchronous pauses) is not represented.

### 2.6 Shared-gain / latent-variable models of noise correlations

**What they answer.** How much of trial-to-trial covariation is one shared gain (multiplicative, Goris, Movshon & Simoncelli 2014), a few latent states (Ecker et al. 2014), or additive plus multiplicative shared factors (Lin et al. 2015). "Population coupling" (Okun et al. 2015) summarizes each neuron's coupling to the summed population rate.

**Strength relative to us.** These models explicitly target **trial-to-trial** shared fluctuations, for example reach vigor or arousal. Our h(t) is identical across reaches, so it does not absorb them. Any per-reach shared gain therefore *could* end up in our J. Our synthetic check found no false coupling gain when a per-reach gain fluctuation was added with J = 0 (TIMEVARYING_FIELD_ISING_REPORT, Validation). That is reassuring but was tested at one gain size only. A real-data control is listed in §4 and §5.

**Weakness relative to us.** A rank-1 gain cannot express *which pairs* are coupled or co-silence structure. Population coupling is one number per neuron, not a pairwise matrix and not a pattern distribution.

### 2.7 Information-theoretic measures and criticality

**Synergy, redundancy and PID.** These ask whether information about a variable (here, reach kinematics) is carried redundantly, uniquely or synergistically by neurons (Schneidman, Bialek & Berry 2003; Williams & Beer 2010). They need a target variable and many samples per state, and PID depends on the choice of redundancy measure. *Our multi-information I(t) is different:* it measures how far the population's patterns are from independent at time t, with no target variable. It is the total correlation of the model distribution, not information about the reach.

**Criticality and thermodynamics.**
- Mora & Bialek (2011) and Tkačik et al. (2015) fit a stationary maxent model and then scale its energy by an artificial temperature. A heat-capacity peak near T = 1, sharpening with N, is read as a sign of criticality.
- Known pitfall: latent common inputs and subsampling produce these signatures without any fine-tuning (Schwab, Nemenman & Mehta 2014; Aitchison, Corradi & Latham 2016; Nonnenmacher et al. 2017).
- Our old pipeline ran temperature sweeps (`old_code/energy_over_time_script/model.py`, `heat_capacity`), and our criticality proxy did not replicate (RESULTS_LOG D).
- **Our current "thermodynamic" quantities are a different use of the same mathematics.** We never change the temperature. At the fitted temperature (T = 1), in each 10-ms bin, we compute the model's entropy S(t), multi-information I(t), coupling susceptibility χ_J(t) = Σ_{i≠j} Cov_t(s_i, s_j) and heat capacity C(t) = Var_t(E). These are descriptive statistics of the fitted distribution at that moment, not criticality claims. The paper should say so explicitly to avoid inheriting the criticality controversy.

---

## 3. What the thermodynamic approach adds (tied to our results)

### 3.1 The model in one line

P_t(s) ∝ exp( Σ_i h_i(t) s_i + ½ Σ_{i≠j} J_ij s_i s_j ),  s_i ∈ {−1, +1},  h_i(t) = cubic B-splines (`thermo_model.py`).

The fit is exact maximum likelihood by enumerating all 2^N patterns. At the optimum, the model's per-bin firing probabilities equal the data's PSTHs (smoothed by the splines), and the model's pairwise correlations, pooled over time, equal the data's. J is therefore what is left of the pairwise structure **after** every neuron's reach-locked rate has been matched in every bin.

### 3.2 What is unique, and the result each feature produced

| Feature | What it enables | Result | Could another method have produced it? |
|---|---|---|---|
| **Exact, normalized joint likelihood for N ≤ 20** | Fair held-out comparisons of *nested generative models of whole population patterns*: M0 (time-varying independent), Ms (static Ising), M1 (time-varying fields + fixed J) | HANDOFF §7.3, exact, 13 sessions: in the ±200-ms reach window, M0 beats Ms in 11/13 (p = 0.15); at baseline Ms is better; M1 beats M0 in 12/13 (p = 0.0005). Also A4 (21/21 sessions, +0.21 nats/bin) and B10 | **Partly.** A GLM gives *conditional* likelihoods per neuron, which do not add up to a joint pattern likelihood. Ranking "rates vs couplings vs both" in one currency (nats per population pattern) and per time window needs a normalized joint model. Correlation methods have no likelihood |
| **Time-varying fields absorbing reach-locked rates** | J is measured beyond reach-locked rates | **A7:** J improves held-out prediction in 21/21 sessions, 63/63 conditions above the trial-shuffle null (+0.26 nats/bin); exact likelihood agrees 38/39; survives pruning pairs with r > 0.5. **A8:** no detected change during the reach (acceleration −15%, 95% CI −28% to +93%) | **Largely yes, with a GLM.** A logistic GLM with a spline PSTH baseline and *same-bin* coupling is our pseudo-likelihood M1 and would find coupling beyond rates. A shift-predictor-corrected JPSTH would show per-pair time-locked noise covariance and could test constancy pair by pair, with low power. A *lagged-only* GLM would miss most of it (Mlag: ~¼ of the gain). GPFA/FA would show shared variance but could not say it is fast and pairwise. **What we add:** one symmetric J that also reproduces higher-order statistics (P(K); A2, HANDOFF §7.1), exact likelihood, and the same J reused in the thermodynamic and κ analyses |
| **Diagnosing the stationary-model artefact** | Same data, two nested models: shows where rate leaks into coupling | **B2 vs B2 note; B3; D table:** "J rises during the reach" and "the energy drop is carried by coupling" were stationary-model artefacts. With h(t), the field term carries the drop (20/21, p_FDR = 2e-5) and the coupling term is ~20× smaller | A shift-predictor JPSTH would never have produced the artefact in the first place. The *lesson* (Tyrcha 2013) is known. Our contribution is showing it quantitatively in PCs, as a caution for maxent users |
| **Per-bin field and coupling energy** | Splits the energy of the observed pattern at each bin into −Σh_i(t)s_i and −½ΣJ_ij s_i s_j | **HANDOFF §7.4:** the field term troughs at a median −40 ms from peak velocity (= median peak acceleration). It falls in the 200 ms before peak velocity in 60% of conditions vs 14% of control windows; depth 15 vs 3.6 baseline-SD; 20/21 sessions, p = 2e-6. The population-average rate barely moves | **Mostly yes, by a rate method that respects sign.** Because increasers and decreasers cancel, the *mean* rate misses it, but the norm of the PSTH-vector change, |Δ rate| summed over cells, or PC1/dPCA of the PSTHs would very likely show an excursion at the same time. GPFA would show a latent trajectory peak. The field energy adds a log-odds weighting (cells near p = 0 or 1 count more) and puts the signal in the same units as the coupling energy, which is what lets us say "movement is in the fields, not in J". **We should test this directly (§5) before calling the dip unique** |
| **Per-bin multi-information I(t), coupling susceptibility χ_J(t), heat capacity C(t)** | A time-resolved readout of *how much the fixed couplings shape population patterns now*. This changes even with J fixed, because it depends on where each cell sits (p) | E10 H2–H4 (pre-registered; step 2 running on the cluster; **no real-data result yet**). Validation: ΔI/N correct in sign 40/40 and within 50% of truth 40/40; shuffle null |ΔI| ≤ 2e-4 (H0, prereg) | **No.** Correlation methods measure *expressed* correlation, which mixes coupling with rate (de la Rocha 2007). GLMs could simulate this but do not normally report a per-bin population quantity. Latent models have no coupling term. This is the clearest unique contribution |
| **Operating-point decomposition** | With J fixed, covariance scales with the variances of the cells involved. To first order, Cov_t(s_i, s_j) ≈ J_ij · Var_t(s_i) · Var_t(s_j), with Var(s_i) = 4p_i(1 − p_i). Cells moving away from p = 0.5 make the same J matter less | **H1 (HANDOFF §8):** reach vs baseline, 60% of 1092 neurons move *away* from 0.5; session-mean p(1 − p) falls in 17/21 sessions (p = 0.018). Fixed J therefore predicts that coupling influence *falls* during acceleration, which is consistent with B10 (Ising worse than M0 during acceleration, 2/21) and §7.3 | The p(1 − p) change is model-free. Turning it into a quantitative prediction for population structure, and testing it against the empirical time-locked covariance (H3a), needs a generative model with explicit J. This reconciles "synchrony changes during the reach" (Nashef 2023) with "J does not change" (A8): expressed synchrony can change through p alone. **No other method in the table makes that distinction explicitly** |
| **κ test under perturbation** | J_eff = (1 + κ)J in the pulse window, with block-specific fields absorbing the laser-driven rate change; one parameter means usable power at ~17 reaches | E11 synthetic validation passed: R = 18, true Δκ = 0.5 recovered at 0.46–0.48; rate-only scenarios gave Δκ ≈ 0 (−0.007 to −0.005); control κ ≈ 0. E10 step 1: κ̂ = 0.56–0.60 for truth 0.6 and ≈ 0 for truth 0. **Real data running; no result yet.** The prior per-block look (RESULTS_LOG C) does not survive FDR | **Partly.** A coupled GLM could add a "laser × coupling" interaction, and a JPSTH could compare stim vs no-stim, but both are confounded by the laser-driven rate change unless the baseline also changes by block. Our model handles this by construction and adds the thermodynamic readout in the pulse window (S2) |
| **Exact sampling** | Model-based nulls; synchronous-pause statistics with vs without J at matched rates | Pause-tail metric dropped after validation (prereg amendment 4); co-silence remains a candidate (PAPER_NOVELTY_CANDIDATES #1) | Pair methods cannot give "P(≥ k of N cells silent)" for N > 3 without many more trials |

### 3.3 Which findings could have come from other methods (honest list)

- **Could have come from other methods:**
  - collective structure beyond independent firing (A2; model-free P(K));
  - J exceeding rate-preserving surrogates (A3);
  - coupling beyond reach-locked rate (A7; same-bin coupled GLM);
  - no detected reach-related change in coupling (A8; GLM with epoch interaction, or epoch-wise shift-corrected noise correlation, at similar or lower power);
  - neurons moving away from 0.5 (H1; just PSTHs);
  - probably the timing of the field dip (§7.4; signed population-vector norm).
- **Needed the maxent framework:**
  - the joint-likelihood ranking of rate-only vs coupling-only vs both, by time window (§7.3, B10);
  - the per-bin *coupling influence* (I, χ_J) at fixed J, and the operating-point prediction linking rate changes to changes in expressed population structure;
  - the stimulation test in the form "κ with block fields" plus pulse-window thermodynamics. The test itself could be built in a GLM, but not with the same population readout;
  - showing *inside one framework* that the earlier stationary maxent claims were artefacts.

---

## 4. Limitations and how to address them

| Limitation | Why it matters | How to address it |
|---|---|---|
| **N ≤ ~20 for exact enumeration** (13 sessions / 39 conditions exact; others use 16-cell subsets) | I(t) and χ_J grow with N; subsets may miss structure that spans the population | Already pre-registered: 10 random 16-cell subsets for N > 20, checked against full exact fits in the four N = 17–19 sessions (amendment 1; E11 also runs one full N = 18 exactness check). Report per-neuron quantities and compare Δ only within the same N. Optional: MCMC or thermodynamic-integration fits for the N = 21–32 sessions as a sensitivity analysis |
| **10-ms binarization** | Loses ms synchrony, the timescale of Nashef 2023 (1 ms, Gaussian SD 3 ms) and de Solages 2008. 21% of non-empty bins contain ≥ 2 spikes. At ~0.6 spike probability per bin, PCs sit in a regime where binarization saturates | **Bin-size sweep at 5 and 20 ms** (and 2 ms if sparsity allows) of the M1 − M0 held-out gain, A8's constancy test and ΔI. Also compute Nashef's 1-ms SI and a dithered-surrogate SI (Louis 2010) on the *same* reaches to show the ms-synchrony result alongside ours. Optional: count-valued fields (a 3-state model: 0, 1, ≥ 2 spikes) as a robustness check |
| **Pairwise, same-bin, symmetric J** | No direction, no lag structure, no higher-order terms | Mlag already shows that lag explains a minority. Add a **one-bin-lag coupled GLM** (§5). P(K) agreement (HANDOFF §7.1) argues that higher-order terms are not needed at this N |
| **Statistical, not anatomical, coupling** | PCs rarely contact each other directly (collaterals exist; Witter et al. 2016). J most likely reflects shared parallel-fiber, climbing-fiber or interneuron input (Heck 2007; Wise 2010). Common input generates apparent coupling and higher-order structure (Macke, Opper & Bethge 2011; Kulkarni & Paninski 2007) | State this plainly in the paper. Test predictions that follow from shared input: J vs distance between units on the probe, or on-beam vs off-beam (`cellData.channel/depth`, HANDOFF §9.4); J vs complex-spike co-occurrence if CS are sorted (PAPER_NOVELTY_CANDIDATES #4) |
| **Trial-to-trial shared gain not in h(t)** | Reach vigor or arousal fluctuations can load into J | Fit **M1 + per-reach shared offset** (one scalar g_r added to all fields in reach r, à la Goris 2014 / Lin 2015) and check that the J gain survives. Synthetic validation with J = 0 plus a per-reach gain already gave gain ≤ 0 |
| **Small trial counts** (9–60 reaches; median ≈ 17 per condition) | Per-pair epoch J is underpowered (amendment 2); confidence intervals are wide (A8: −28% to +93%) | Single-parameter κ; session-level statistics; equivalence bounds for any "no change" claim (E11 S1 uses |κ| < 0.3). Report CIs, not just p-values |
| **Pseudo-likelihood fails moment matching in strongly coupled sessions** (2022 cohort, e.g. 220516) | Thermodynamic quantities from pseudo-likelihood parameters would be wrong | Already the rule: exact ML for all thermodynamics; pseudo-likelihood only for initialization and for the A7 held-out gain, which exact likelihood confirmed (38/39) |
| **Fields smooth on 50-ms knots** | Rate changes faster than 50 ms leak into J or are blurred | Knot-spacing sensitivity (25 / 50 / 100 ms) on the A7 gain |
| **Fixed J by design** | Cannot see J(t) | κ in windows; per-pair epoch J as secondary; state-space Ising (Shimazaki 2012) is a possible future comparison if trial counts allow |

---

## 5. Recommended text, figure and benchmark analyses for the paper

### 5.1 Suggested methods-comparison paragraph (Discussion or Methods)

> Coordinated Purkinje-cell activity has mostly been studied with pairwise synchrony measures at millisecond resolution, such as cross-correlograms and synchrony indices (de Solages et al., 2008; Heck et al., 2007; Sedaghat-Nejad et al., 2022; Nashef et al., 2023). These measures are model-free and temporally precise, but when cells share movement-related rate changes, rate co-modulation must be removed by shuffle predictors, and the measured synchrony still depends on firing rate. Latent-variable methods (e.g. GPFA; Yu et al., 2009) capture shared population dynamics but do not separate fast pairwise interaction from shared drive. We used a pairwise maximum-entropy model in which each neuron's field follows its reach-locked rate while couplings are fixed, extending stimulus-dependent maximum-entropy models (Granot-Atedgi et al., 2013). Because the model is fitted by exact likelihood, it allowed held-out comparison of nested models and, at every 10-ms bin, measurement of how strongly the fixed couplings shape population patterns (multi-information, coupling susceptibility). The same framework showed that a stationary model attributes movement-related rate changes to coupling (Tyrcha et al., 2013), an artefact we observed directly. Couplings beyond reach-locked rates were present at rest, and we found no evidence that they change during the reach. Because the model is fitted on 10-ms binarized activity, it does not test the millisecond synchrony reported by Nashef et al. (2023). It also measures statistical rather than synaptic coupling, most likely reflecting shared input.

(Edit the last two sentences once E10 and E11 results are in.)

### 5.2 Figure concept: "one dataset, five lenses"

One supplementary figure (or main figure 1b), using a single example session and the session-level summary:

- **a. Shift-predictor-corrected JPSTH / SI** for an example pair (1-ms and 10-ms bins): what the standard method shows.
- **b. Stationary Ising:** the coupling energy appears to rise during the reach (the artefact, B2/B3).
- **c. Time-varying-field model:** the drop moves into the field term; held-out J gain is flat across epochs (A7/A8).
- **d. Benchmarks on the same folds:**
  - coupled GLM with lag-0 vs lag-1 coupling (expected: lag-0 ≈ our M1, lag-1 much smaller);
  - GPFA latent on the same reaches (does a latent capture the field-dip timing?).
- **e. Only in our framework:** I(t)/N, χ_J(t)/N and p(1 − p)(t), with the empirical time-locked covariance overlaid (H3a).
- **Side panel:** a question × method check-mark matrix (a reduced version of the table in §1).

### 5.3 Analyses that would strengthen the comparison (in priority order)

1. **Coupled-GLM benchmark on identical data and folds.** Logistic GLM per neuron with the same B-spline baseline:
   - (i) lag-0 coupling, which should reproduce M1;
   - (ii) 1-bin and 2-bin lagged coupling only;
   - (iii) own-history terms.

   This tells reviewers exactly what part of A7 a standard GLM recovers, and it pre-empts the "isn't this just a GLM?" objection. Cheap: the logistic machinery already exists in `tvfield_model.py`.
2. **Bin-size sweep (5 / 10 / 20 ms; 2 ms if possible)** of the M1 − M0 gain, the epoch-constancy test and ΔI/N, plus Nashef's 1-ms SI with a dithering null on the same reaches. This addresses the most likely reviewer question: does coupling at ms timescales change with the reach, even if 10-ms coupling does not?
3. **Shared-gain control.** M1 plus one per-reach additive field offset (or a rank-1 latent). If the J gain survives, trial-to-trial shared excitability is ruled out as the source.
4. **Rate-method test of the field dip.** Compute the norm of the PSTH-vector change, summed |Δp_i| and dPCA/PC1 timing, using the same control-window procedure. If they match the dip timing, present the dip as a rate result that the energy framework *expresses* in common units. If they do not, the log-odds weighting is doing real work and the claim gets stronger.
5. **Subset validation** (pre-registered): Spearman ρ and sign agreement of subset vs full ΔI/N in the N = 17–19 sessions.
6. **Shared-input simulation.** Generate PC-like spike trains from a common-input model with *no* direct coupling, at our rates and N, and show that the method returns J > 0 with the same properties. This turns the "statistical, not anatomical" caveat into a stated interpretation.
7. **Spatial structure.** J vs inter-unit distance and on-beam alignment, a direct link to Heck et al. (2007).

---

## 6. Verified bibliography

All DOIs were checked against Crossref (title, first author, year, journal) on 2026-10-12. Preprints are marked.

**PC synchrony and cerebellum**
- de Solages C, Szapiro G, Brunel N, et al. (2008). High-frequency organization and synchrony of activity in the Purkinje cell layer of the cerebellum. *Neuron* 58:775–788. doi:10.1016/j.neuron.2008.05.008
- Heck DH, Thach WT, Keating JG (2007). On-beam synchrony in the cerebellum as the mechanism for the timing and coordination of movement. *PNAS* 104:7658–7663. doi:10.1073/pnas.0609966104
- Wise AK, Cerminara NL, Marple-Horvat DE, Apps R (2010). Mechanisms of synchronous activity in cerebellar Purkinje cells. *J Physiol* 588:2373–2390. doi:10.1113/jphysiol.2010.189704
- Person AL, Raman IM (2012; online 2011). Purkinje neuron synchrony elicits time-locked spiking in the cerebellar nuclei. *Nature* 481:502–505. doi:10.1038/nature10732
- Ramirez JE, Stell BM (2016). Calcium imaging reveals coordinated simple spike pauses in populations of cerebellar Purkinje cells. *Cell Rep* 17:3125–3132. doi:10.1016/j.celrep.2016.11.075
- Sedaghat-Nejad E, Pi JS, Hage P, Fakharian MA, Shadmehr R (2022). Synchronous spiking of cerebellar Purkinje cells during control of movements. *PNAS* 119:e2118954119. doi:10.1073/pnas.2118954119
- Nashef A, Spindle MS, Calame DJ, Person AL (2023). A dual Purkinje cell rate and synchrony code sculpts reach kinematics. *bioRxiv* (preprint; PMID 37503038, PMC10370034). doi:10.1101/2023.07.12.548720
- Herzfeld DJ, Kojima Y, Soetedjo R, Shadmehr R (2015). Encoding of action by the Purkinje cells of the cerebellum. *Nature* 526:439–442. doi:10.1038/nature15693
- Witter L, Rudolph S, Pressler RT, Lahlaf SI, Regehr WG (2016). Purkinje cell collaterals enable output signals from the cerebellar cortex to feed back to Purkinje cells and interneurons. *Neuron* 91:312–319. doi:10.1016/j.neuron.2016.05.037
- Chen S, Augustine GJ, Chadderton P (2016). The cerebellum linearly encodes whisker position during voluntary movement. *eLife* 5:e10509. doi:10.7554/eLife.10509
- Hong S, Negrello M, Junker M, et al. (2016). Multiplexed coding by cerebellar Purkinje neurons. *eLife* 5:e13810. doi:10.7554/eLife.13810
- Calame DJ, Becker MI, Person AL (2023). Cerebellar associative learning underlies skilled reach adaptation. *Nat Neurosci* 26:1068–1079. doi:10.1038/s41593-023-01347-y (dataset; preprint doi:10.1101/2021.12.17.473247)
- Becker MI, Person AL (2019). Cerebellar control of reach kinematics for endpoint precision. *Neuron* 103:335–348. doi:10.1016/j.neuron.2019.05.007

**Pairwise correlation, predictors, unitary events**
- Perkel DH, Gerstein GL, Moore GP (1967). Neuronal spike trains and stochastic point processes. II. Simultaneous spike trains. *Biophys J* 7:419–440. doi:10.1016/S0006-3495(67)86597-4
- Gerstein GL, Perkel DH (1972). Mutual temporal relationships among neuronal spike trains. *Biophys J* 12:453–473. doi:10.1016/S0006-3495(72)86097-1
- Aertsen AM, Gerstein GL, Habib MK, Palm G (1989). Dynamics of neuronal firing correlation: modulation of "effective connectivity". *J Neurophysiol* 61:900–917. doi:10.1152/jn.1989.61.5.900
- Brody CD (1999). Correlations without synchrony. *Neural Comput* 11:1537–1551. doi:10.1162/089976699300016133
- Grün S, Diesmann M, Aertsen A (2002). Unitary events in multiple single-neuron spiking activity: I. Detection and significance. *Neural Comput* 14:43–80. doi:10.1162/089976602753284455
- Ventura V, Cai C, Kass RE (2005). Trial-to-trial variability and its effect on time-varying dependency between two neurons. *J Neurophysiol* 94:2928–2939. doi:10.1152/jn.00644.2004
- Louis S, Gerstein GL, Grün S, Diesmann M (2010). Surrogate spike train generation through dithering in operational time. *Front Comput Neurosci* 4:127. doi:10.3389/fncom.2010.00127
- de la Rocha J, Doiron B, Shea-Brown E, Josić K, Reyes A (2007). Correlation between neural spike trains increases with firing rate. *Nature* 448:802–806. doi:10.1038/nature06028
- Cohen MR, Kohn A (2011). Measuring and interpreting neuronal correlations. *Nat Neurosci* 14:811–819. doi:10.1038/nn.2842
- Cutts CS, Eglen SJ (2014). Detecting pairwise correlations in spike trains: an objective comparison of methods and application to the study of retinal waves. *J Neurosci* 34:14288–14303. doi:10.1523/JNEUROSCI.2767-14.2014

**GLMs and connectivity inference**
- Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005). A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects. *J Neurophysiol* 93:1074–1089. doi:10.1152/jn.00697.2004
- Okatan M, Wilson MA, Brown EN (2005). Analyzing functional connectivity using a network likelihood model of ensemble neural spiking activity. *Neural Comput* 17:1927–1961. doi:10.1162/0899766054322973
- Pillow JW, Shlens J, Paninski L, et al. (2008). Spatio-temporal correlations and visual signalling in a complete neuronal population. *Nature* 454:995–999. doi:10.1038/nature07140
- Kulkarni JE, Paninski L (2007). Common-input models for multiple neural spike-train data. *Network* 18:375–407. doi:10.1080/09548980701625173
- Vidne M, Ahmadian Y, Shlens J, et al. (2012). Modeling the impact of common noise inputs on the network activity of retinal ganglion cells. *J Comput Neurosci* 33:97–121. doi:10.1007/s10827-011-0376-2
- Stevenson IH, Rebesco JM, Miller LE, Körding KP (2008). Inferring functional connections between neurons. *Curr Opin Neurobiol* 18:582–588. doi:10.1016/j.conb.2008.11.005
- Das A, Fiete IR (2020). Systematic errors in connectivity inferred from activity in strongly recurrent networks. *Nat Neurosci* 23:1286–1296. doi:10.1038/s41593-020-0699-2

**Maximum entropy / Ising**
- Schneidman E, Berry MJ, Segev R, Bialek W (2006). Weak pairwise correlations imply strongly correlated network states in a neural population. *Nature* 440:1007–1012. doi:10.1038/nature04701
- Shlens J, Field GD, Gauthier JL, et al. (2006). The structure of multi-neuron firing patterns in primate retina. *J Neurosci* 26:8254–8266. doi:10.1523/JNEUROSCI.1282-06.2006
- Roudi Y, Nirenberg S, Latham PE (2009). Pairwise maximum entropy models for studying large biological systems: when they can work and when they can't. *PLoS Comput Biol* 5:e1000380. doi:10.1371/journal.pcbi.1000380
- Roudi Y, Hertz J (2011). Mean field theory for nonequilibrium network reconstruction. *Phys Rev Lett* 106:048702. doi:10.1103/PhysRevLett.106.048702
- Macke JH, Opper M, Bethge M (2011). Common input explains higher-order correlations and entropy in a simple model of neural population activity. *Phys Rev Lett* 106:208102. doi:10.1103/PhysRevLett.106.208102
- Aurell E, Ekeberg M (2012). Inverse Ising inference using all the data. *Phys Rev Lett* 108:090201. doi:10.1103/PhysRevLett.108.090201
- Shimazaki H, Amari S, Brown EN, Grün S (2012). State-space analysis of time-varying higher-order spike correlation for multiple neural spike train data. *PLoS Comput Biol* 8:e1002385. doi:10.1371/journal.pcbi.1002385
- Granot-Atedgi E, Tkačik G, Segev R, Schneidman E (2013). Stimulus-dependent maximum entropy models of neural population codes. *PLoS Comput Biol* 9:e1002922. doi:10.1371/journal.pcbi.1002922
- Tyrcha J, Roudi Y, Marsili M, Hertz J (2013). The effect of nonstationarity on models inferred from neural data. *J Stat Mech* 2013:P03005. doi:10.1088/1742-5468/2013/03/P03005
- Tkačik G, Marre O, Amodei D, et al. (2014). Searching for collective behavior in a large network of sensory neurons. *PLoS Comput Biol* 10:e1003408. doi:10.1371/journal.pcbi.1003408
- Donner C, Obermayer K, Shimazaki H (2017). Approximate inference for time-varying interactions and macroscopic dynamics of neural populations. *PLoS Comput Biol* 13:e1005309. doi:10.1371/journal.pcbi.1005309
- Gardella C, Marre O, Mora T (2019). Modeling the correlated activity of neural populations: a review. *Neural Comput* 31:233–269. doi:10.1162/neco_a_01154

**Dimensionality reduction and latent models**
- Yu BM, Cunningham JP, Santhanam G, et al. (2009). Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity. *J Neurophysiol* 102:614–635. doi:10.1152/jn.90941.2008
- Cunningham JP, Yu BM (2014). Dimensionality reduction for large-scale neural recordings. *Nat Neurosci* 17:1500–1509. doi:10.1038/nn.3776
- Kobak D, Brendel W, Constantinidis C, et al. (2016). Demixed principal component analysis of neural population data. *eLife* 5:e10989. doi:10.7554/eLife.10989
- Elsayed GF, Cunningham JP (2017). Structure in neural population recordings: an expected byproduct of simpler phenomena? *Nat Neurosci* 20:1310–1318. doi:10.1038/nn.4617
- Pandarinath C, O'Shea DJ, Collins J, et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nat Methods* 15:805–815. doi:10.1038/s41592-018-0109-9
- Semedo JD, Zandvakili A, Machens CK, Yu BM, Kohn A (2019). Cortical areas interact through a communication subspace. *Neuron* 102:249–259. doi:10.1016/j.neuron.2019.01.026

**Shared gain and population coupling**
- Goris RLT, Movshon JA, Simoncelli EP (2014). Partitioning neuronal variability. *Nat Neurosci* 17:858–865. doi:10.1038/nn.3711
- Ecker AS, Berens P, Cotton RJ, et al. (2014). State dependence of noise correlations in macaque primary visual cortex. *Neuron* 82:235–248. doi:10.1016/j.neuron.2014.02.006
- Okun M, Steinmetz NA, Cossell L, et al. (2015). Diverse coupling of neurons to populations in sensory cortex. *Nature* 521:511–515. doi:10.1038/nature14273
- Lin I-C, Okun M, Carandini M, Harris KD (2015). The nature of shared cortical variability. *Neuron* 87:644–656. doi:10.1016/j.neuron.2015.06.035

**Information theory and criticality**
- Schneidman E, Bialek W, Berry MJ (2003). Synergy, redundancy, and independence in population codes. *J Neurosci* 23:11539–11553. doi:10.1523/JNEUROSCI.23-37-11539.2003
- Williams PL, Beer RD (2010). Nonnegative decomposition of multivariate information. *arXiv*:1004.2515 (preprint)
- Mora T, Bialek W (2011). Are biological systems poised at criticality? *J Stat Phys* 144:268–302. doi:10.1007/s10955-011-0229-4
- Schwab DJ, Nemenman I, Mehta P (2014). Zipf's law and criticality in multivariate data without fine-tuning. *Phys Rev Lett* 113:068102. doi:10.1103/PhysRevLett.113.068102
- Tkačik G, Mora T, Marre O, et al. (2015). Thermodynamics and signatures of criticality in a network of neurons. *PNAS* 112:11508–11513. doi:10.1073/pnas.1514188112
- Aitchison L, Corradi N, Latham PE (2016). Zipf's law arises naturally when there are underlying, unobserved variables. *PLoS Comput Biol* 12:e1005110. doi:10.1371/journal.pcbi.1005110
- Nonnenmacher M, Behrens C, Berens P, Bethge M, Macke JH (2017). Signatures of criticality arise from random subsampling in simple population models. *PLoS Comput Biol* 13:e1005718. doi:10.1371/journal.pcbi.1005718

### Notes on the literature search
- "None found" / "we did not find" in the table means our searches (Crossref, web) turned up no example. It is not proof of absence. In particular, a coupled-GLM or maxent analysis of PC populations may exist, so run one more targeted PubMed search before the paper says "first".
- Nashef et al. (2023) was still listed only as a bioRxiv preprint in our search. Check for a journal version before submission.
