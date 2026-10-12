# Literature context for the coupling-term (J) extraction method

Two questions: (1) has the *field-vs-coupling energy decomposition* used in this
project's pipeline (splitting a fitted Ising Hamiltonian into a local-field term
and a pairwise-coupling term, then tracking the coupling term as its own signal)
been explored in the statistical-mechanics/thermodynamics literature outside
neuroscience, and (2) what is the current landscape of neuroscience methods for
detecting correlated firing activity, of which the pairwise maximum-entropy
(Ising) approach is one member.

## 1. Statistical-mechanics literature on the field/coupling decomposition

### Structural property vs. citable discovery

The additive split `E(s) = -Σ h_i s_i - Σ_{i<j} J_ij s_i s_j` is built into the
pairwise Ising Hamiltonian itself: it dates to the original lattice model and
is standard in every statistical-mechanics/condensed-matter treatment of spin
systems with an external field plus exchange interactions (textbook material,
not a citable "finding"). What *is* a distinct, citable research thread is:
(a) how the coupling parameters `J_ij` are estimated from data (the "inverse
Ising problem"), and (b) using the fitted field/coupling split as an analytic
tool to ask how much of a system's structure is collective (pairwise-dependent)
versus explainable by single-unit statistics alone, which is exactly what this
project's `calc_e_with_terms` / P(K) analysis does.

### Origin: maximum entropy for many-body data

The general framework — find the least-structured (maximum-entropy) distribution
consistent with observed first and second moments, which is provably the pairwise
Ising/Boltzmann form — traces to Jaynes' maximum-entropy principle and was first
applied to real multi-unit neural data by [Schneidman, Berry & Segev 2006](https://doi.org/10.1038/nature04701),
which showed that weak pairwise correlations, propagated through such a fitted
model, imply strongly correlated network-wide activity patterns in a retinal
population: the founding demonstration that the pairwise-coupling term captures
genuine collective structure the independent (rate-only) model misses.

### Solving the inverse problem (estimating h, J from data)

A large statistical-physics literature addresses how to fit `(h,J)` efficiently,
spanning exact/near-exact Monte-Carlo-based learning (the approach this project
uses, via ConIII's MCH solver) and faster approximate solvers:
- [Sessak & Monasson 2009](https://doi.org/10.1088/1751-8113/42/5/055001): small-correlation expansion for the inverse Ising problem.
- [Nguyen & Berg 2012](https://doi.org/10.1088/1742-5468/2012/03/p03004): Bethe–Peierls approximation.
- [Roudi & Hertz 2011](https://doi.org/10.1088/1742-5468/2011/03/p03031) and [Mézard & Sakellariou 2011](https://doi.org/10.1088/1742-5468/2011/07/l07001): TAP/mean-field solutions for (kinetic) Ising inference.
- [Barton, De Leonardis, Coucke & Cocco 2016](https://doi.org/10.1093/bioinformatics/btw328): ACE, adaptive cluster expansion for maximum-entropy graphical models.
- [Lokhov, Vuffray, Misra & Chertkov 2018](https://doi.org/10.1126/sciadv.1700791): provably optimal structure/parameter learning for Ising models.
- [Cocco, Feinauer, Figliuzzi, Monasson & Weigt 2017](https://doi.org/10.1088/1361-6633/aa9965): a broad review of inverse-Ising/maxent methods across biological data (protein sequences, but the methodological survey covers the same field/coupling estimation machinery used here).

### The field/coupling split as an analytic tool

The step this project takes beyond simply fitting `(h,J)` — separating a
configuration's energy into its field contribution and its coupling contribution
to build a coupling-specific time series, and asking whether the coupling term
alone (not firing rate) tracks a signal of interest, and has direct precedent in a
handful of papers that use the same fitted-model decomposition as a diagnostic:
- [Tkačik, Marre, Amodei, Schneidman, Bialek & Berry 2014](https://doi.org/10.1371/journal.pcbi.1003408), "Searching for Collective Behavior in a Large Network of Sensory Neurons" explicitly compares a population's entropy/energy under the independent model against the pairwise (coupling-included) model to quantify how much structure is genuinely collective.
- [Yu, Yang, Nakahara, Santos, Nikolić & Plenz 2011](https://doi.org/10.1523/jneurosci.3127-11.2011), "Higher-Order Interactions Characterized in Cortical Activity" decomposes population statistics into pairwise and higher-order contributions to isolate the coupling-driven structure from independent firing.
- [Ruffini, Damiani, Lozano-Soldevilla et al. 2023](https://doi.org/10.1371/journal.pcbi.1010811) fit an Ising model to brain-activity data and track a derived "Ising temperature": a related use of the fitted-model machinery (rather than raw firing rate) as the signal of interest.
- [Olsen, Whitlock & Roudi 2024](https://doi.org/10.1371/journal.pcbi.1012074), "The quality and complexity of pairwise maximum entropy models for large cortical populations," examines when the fitted coupling term is even a reliable descriptor as population size grows: relevant to how much to trust a J-derived signal at the population sizes used here.

**Bottom line for part 1:** no paper treats "splitting Ising energy into field vs.
coupling contributions" as a named method in its own right in pure thermodynamics.
It is an immediate consequence of the Hamiltonian's linear structure. What the
literature does establish, repeatedly, is that the *coupling term specifically*
(as opposed to the field term or raw activity/firing rate) is the quantity that
carries information about genuine collective structure, and several groups use
exactly this field/coupling split as an analytic device once `(h,J)` is fit,
which is the closest documented ancestor to this project's own use of J as an
independent signal.

## 2. Current neuroscience methods for identifying correlated firing activity

The maximum-entropy/Ising approach is one branch of a broader toolkit. Grouped
by strategy:

**Pairwise/bivariate correlation measures.** The baseline approach: cross-correlograms
and correlation-coefficient estimates between spike trains. Their pitfalls
(binning, rate confounds, finite-sample bias) are systematically compared in
[Cutts & Eglen 2014](https://doi.org/10.1523/jneurosci.2767-14.2014). Because
raw correlation coefficients depend strongly on firing rate and window choice,
follow-up work such as [Yatsenko, Josić, Ecker et al. 2015](https://doi.org/10.1371/journal.pcbi.1004083)
addresses improved estimation and interpretation of these correlations. Broader
tutorial treatment of correlation-based functional-connectivity analysis (and its
interpretational hazards) is given in [Bastos & Schoffelen 2016](https://doi.org/10.3389/fnsys.2015.00175).

**Exact-synchrony / pattern-detection methods.** Rather than a smooth correlation
statistic, these ask whether specific spike patterns (near-simultaneous firing
across many neurons) occur more often than chance. Unitary Events-style analyses
and their extensions include [Pipa, Wheeler, Singer & Grün 2008](https://doi.org/10.1007/s10827-007-0065-3) ("NeuroXidence"),
[Torre, Picado-Muiño, Denker et al. 2013](https://doi.org/10.3389/fncom.2013.00132) (frequent-itemset mining for synchronous patterns),
[Torre, Canova, Denker et al. 2016](https://doi.org/10.1371/journal.pcbi.1004939) ("ASSET," for sequences of synchronous events),
and [Quaglio, Yegenoglu, Torre et al. 2017](https://doi.org/10.3389/fncom.2017.00041) ("SPADE," spatio-temporal pattern detection).

**Maximum-entropy / Ising population models.** The approach this project uses:
fit a pairwise (or higher-order) maximum-entropy model and read the coupling
terms as the correlation structure. Beyond the foundational
[Schneidman et al. 2006](https://doi.org/10.1038/nature04701) paper, this
includes higher-order extensions ([Yu et al. 2011](https://doi.org/10.1523/jneurosci.3127-11.2011);
[Köster, Sohl-Dickstein, Gray & Olshausen 2014](https://doi.org/10.1371/journal.pcbi.1003684))
and scaling studies of when pairwise models remain adequate as population size
grows ([Tkačik et al. 2014](https://doi.org/10.1371/journal.pcbi.1003408);
[Olsen, Whitlock & Roudi 2024](https://doi.org/10.1371/journal.pcbi.1012074)).

**Model-based coupling via GLMs.** An alternative to the Ising/energy-based
framework: fit a generalized linear (point-process) model to each neuron's
spike train with explicit "coupling filters" from other neurons' recent spiking,
so functional coupling is a fitted filter rather than an Ising `J`. The landmark
application is [Pillow, Shlens, Paninski et al. 2008](https://doi.org/10.1038/nature07140),
"Spatio-temporal correlations and visual signalling in a complete neuronal
population," with related population-level GLM formulations reviewed in
Macke, Buesing & Cunningham (2011), "Empirical models of spiking in neural populations."

**Population-level summary measures.** Rather than every pairwise term, some
methods summarize how coupled each neuron is *to the rest of the population* as
a single number. [Okun, Steinmetz, Cossell et al. 2015](https://doi.org/10.1038/nature14273),
"Diverse coupling of neurons to populations in sensory cortex" is the key
reference, and is conceptually the closest published analogue to a per-neuron
"coupling strength" score.

**Noise-correlation frameworks.** A large body of work treats "correlated
activity" specifically as shared trial-to-trial variability around a mean
response, distinct from stimulus-driven co-activation. See
[Lin, Okun, Carandini & Harris 2015](https://doi.org/10.1016/j.neuron.2015.06.035),
"The Nature of Shared Cortical Variability," and
[Hansen, Chelaru & Dragoi 2012](https://doi.org/10.1016/j.neuron.2012.08.029),
"Correlated Variability in Laminar Cortical Circuits."

**Latent-variable / dimensionality-reduction approaches.** Instead of modeling
pairwise structure directly, these extract shared low-dimensional population
"modes" (e.g., Gaussian-process factor analysis and its descendants) whose
loadings implicitly encode which neurons co-vary; representative recent uses
include [Gallego, Perich, Naufel et al. 2018](https://doi.org/10.1038/s41467-018-06560-z)
and [Pandarinath, O'Shea, Collins et al. 2017](https://doi.org/10.1101/152884)
(sequential autoencoders for single-trial population dynamics).

**Directed/information-theoretic connectivity measures.** For asking not just
*whether* activity is correlated but which neuron's activity predicts another's
in time. Point-process Granger causality is one route
([Kim, Putrino, Ghosh & Brown 2011](https://doi.org/10.1371/journal.pcbi.1001110);
[Sheikhattar, Miran, Liu et al. 2018](https://doi.org/10.1073/pnas.1718154115))
and transfer entropy
([Ito, Hansen, Heiland et al. 2011](https://doi.org/10.1371/journal.pone.0027431);
[Shorten, Spinney & Lizier 2021](https://doi.org/10.1371/journal.pcbi.1008054)).

### This project's place in the landscape

The pipeline's approach — fitting a pairwise maximum-entropy (Ising) model via
Monte-Carlo Boltzmann learning and reading out the coupling term `J` as a
distinct signal from the local-field/firing-rate term. It sits squarely in the
maximum-entropy branch above, and its specific move of decomposing energy into
field vs. coupling contributions to build a J-only time series has the most
direct precedent in the collective-vs-independent comparisons of
[Tkačik et al. 2014](https://doi.org/10.1371/journal.pcbi.1003408) and
[Yu et al. 2011](https://doi.org/10.1523/jneurosci.3127-11.2011), and is
complementary to — rather than competing with — the GLM-coupling-filter,
population-coupling, and noise-correlation frameworks used elsewhere in the
field for the same underlying question of "which neurons/units are firing
together, and does that co-firing carry information beyond individual rates."


## 3. Comparison with Tkačik et al. 2014

The user's pipeline and [Tkačik, Marre, Amodei, Schneidman, Bialek & Berry 2014](https://doi.org/10.1371/journal.pcbi.1003408)
both fit a pairwise maximum-entropy/Ising model and both examine the
population-synchrony distribution `P(K)` as a check on that model, but the two
diverge on system, model form, and what the fitted model is used to show.

**Biological system and experimental design.** Tkačik et al. record extracellular
activity from salamander retinal ganglion cells (up to 160 cells) responding
to a repeated, passive naturalistic movie: a sensory-encoding paradigm with
hundreds of identical stimulus repeats and no behavior. This project instead
fits the model to cerebellar Purkinje-cell populations recorded in vivo while
a mouse performs a reaching/arm-movement task: a motor-control paradigm with
single-trial kinematics (velocity, acceleration) as the variable of interest,
not a repeated stimulus.

**Model form.** Both start from the same field-plus-coupling energy,
`H = H⁽¹⁾ + H⁽²⁾` with `H⁽¹⁾ = -Σh_i s_i` and `H⁽²⁾ = -½Σ J_ij s_i s_j`. Tkačik
et al. add a third term, `H⁽ᴷ⁾`, with one Lagrange multiplier `λ_K` per possible
population count `K`, to build what they call the "K-pairwise model." They
introduce it because at their scale (N up to 100+) a pure pairwise fit
systematically misses the tail of the population-synchrony distribution
`P(K)`, i.e., how often many cells fire together; the extra term is
constrained to match `P(K)` exactly. This project's model has no `H⁽ᴷ⁾` term:
it fits the plain two-term pairwise energy (`h_i`, `J_ij` only) via ConIII's
MCH solver, at population sizes (tens of Purkinje cells per session) small
enough that the plain pairwise fit already reproduces `P(K)` well enough to
use as a validity check rather than a target requiring its own extra
parameters.

**What `P(K)` is used for.** In Tkačik et al., the mismatch between the
pairwise model's predicted `P(K)` and the data's `P(K)` is itself a central
finding, motivating the K-pairwise extension. In this project's
`pk_j_analysis.py`, the same comparison (correlation of the data's `P(K)`
against the pairwise-model's and the independent-model's predictions) is used
as a binary gate: a session/window is flagged "collective" when the pairwise
model tracks the data's `P(K)` at least as well as the independent model AND
the two models' predicted `P(K)` curves diverge by a set margin. That flag is
then checked against whether a coupling (`J`) peak was detected in the same
window, rather than being used to refine the energy function itself.

**Static/scaling analysis vs. time-resolved, behavior-locked signal.** Tkačik
et al.'s central question is how collective quantities (entropy, multi-information,
a heat-capacity peak under an artificial temperature sweep) scale with population
size `N`, fitting once per stimulus condition over the full recording to ask
whether the network approaches a critical point as more neurons are added.
This project fits `(h,J)` per short time window and builds a **time-resolved
`J(t)` trajectory** across each reach, then asks (in `arbitration_j_many.py`)
whether the largest jump or peak in that `J(t)` trace leads or lags peak
acceleration/velocity of the arm movement, via cross-correlation and lag
statistics computed session by session. Tkačik et al. do not construct a
per-window coupling time series or relate it to any behavioral event; their
unit of analysis is population size, not time.

**Purpose.** Tkačik et al. aim to characterize the intrinsic statistical
structure of a sensory code, independent of any behavioral readout — is the
population's collective behavior extensive, near-critical, becoming more or
less structured as `N` grows. This project's aim is to use the coupling term
as a physiological signal tied to a specific motor event: showing that `J`
transitions co-occur with (and possibly precede) peak kinematic exertion during
a reach, in support of a claim about cerebellar network self-organization
during motor control, rather than a claim about population-scaling or
criticality per se.

**Where the resemblance is genuine.** The energy-decomposition machinery
(splitting `H` into field and coupling terms, treating `si=+1/-1` spike states,
comparing to an independent model, checking `P(K)`) is the same statistical
framework, and Tkačik et al. is the most direct methodological ancestor for
using field/coupling decomposition as a diagnostic (see Part 1 above). The
project's contribution beyond that ancestry is the time-resolved,
single-trial coupling trajectory and its lag/lead comparison against
movement kinematics, which has no equivalent in the 2014 paper.
