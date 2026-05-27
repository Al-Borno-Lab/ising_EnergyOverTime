# Purkinje Cell Population Simulation and Inverse Ising Analysis

## Overview

This document describes the mathematical framework behind the Purkinje cell (PC)
population simulation, the inverse Ising model fitted to the spike data, and a
mechanistic explanation of why the pairwise coupling term **J jumps during the
collective (reach) regime** even when mean firing rates are nearly identical
across regimes.

---

## 1. Biological Motivation

Purkinje cells are the sole output neurons of the cerebellar cortex. During a
mouse arm reach, local field potential recordings and multi-unit spike data show
a detectable change in the **pairwise coupling** extracted by maximum-entropy
(Ising) models, while single-unit firing rates change only modestly. This
simulation is designed to isolate the mechanism: specifically, to demonstrate
that **population synchrony alone** — without any rate change — is sufficient to
produce a J jump.

Two behavioural states are modelled:

| State | Biology | Simulation |
|-------|---------|------------|
| **Random / pre-reach** | PC simple spikes at spontaneous rate, uncorrelated | Independent Poisson at λ_rand ≈ 60 Hz |
| **Collective / during reach** | PC populations co-activate and co-pause in synchrony | Shared burst/quiet Markov state at mean ≈ 56 Hz |

---

## 2. Neural Population Model

### 2.1 Populations

Two populations are simulated:

| Population | Label | Default size | Biology |
|------------|-------|-------------|---------|
| Principal | Purkinje cells | N_e = 100 | GABAergic output cells |
| Interneurons | Basket / Stellate | N_i = 25 | Molecular layer inhibitory cells |

Time is discretised at **dt = 0.1 ms** for a total of **T_total = 500 ms
(5 000 steps)**. Each step t ∈ {0, 1, …, N_t − 1} produces a binary spike
vector for each population.

### 2.2 Regime Schedule

The simulation alternates between blocks of length **L = 200 ms (2 000 steps)**:

```
t:   0 ──────── 200 ms ──────── 400 ms ──────── …
     [  Random  ][  Collective  ][  Random  ][ … ]
```

Formally, the regime indicator is:

$$
m(t) = \begin{cases}
0 & \text{(random)}     & t \bmod 2L < L \\
1 & \text{(collective)} & t \bmod 2L \geq L
\end{cases}
$$

### 2.3 Random Regime — Independent Poisson

In the random blocks each neuron i fires independently at every time step with
probability:

$$
p_{\text{rand}} = \frac{\lambda_{\text{rand}}}{1000} \cdot dt
\quad (\lambda_{\text{rand}} = 60 \text{ Hz}, \; dt = 0.1 \text{ ms})
\quad \Rightarrow \quad p_{\text{rand}} = 6 \times 10^{-3}
$$

The spike variable of neuron i at step t is drawn as:

$$
s_i(t) \sim \mathrm{Bernoulli}(p_{\text{rand}}), \quad \text{i.i.d. across } i
$$

Because each neuron fires **independently**, the pairwise spike probability
factorises:

$$
P(s_i = 1,\; s_j = 1) = p_{\text{rand}}^2
\quad\Longrightarrow\quad
\mathrm{Cov}(s_i, s_j) = 0
$$

This zero-covariance regime is the null model against which the collective regime
is contrasted.

### 2.4 Collective Regime — Shared Burst / Quiet Markov State

During collective blocks a **shared population state** B(t) ∈ {0, 1} evolves
as a two-state continuous-time (discrete-step) Markov chain:

$$
P\bigl(B(t+1) = 1 \mid B(t) = 0\bigr) = p_{\text{on}}  = 0.005
$$
$$
P\bigl(B(t+1) = 0 \mid B(t) = 1\bigr) = p_{\text{off}} = 0.02
$$

The **stationary burst fraction** is:

$$
\phi = \frac{p_{\text{on}}}{p_{\text{on}} + p_{\text{off}}}
    = \frac{0.005}{0.005 + 0.02} = 0.20
$$

Mean sojourn times (geometric distributions):

$$
\tau_{\text{burst}}
  = \frac{1}{p_{\text{off}}} \cdot dt
  = \frac{1}{0.02} \times 0.1 \text{ ms} = 5 \text{ ms}
$$

$$
\tau_{\text{quiet}}
  = \frac{1}{p_{\text{on}}} \cdot dt
  = \frac{1}{0.005} \times 0.1 \text{ ms} = 20 \text{ ms}
$$

Conditioned on B(t), each neuron fires **independently** with rate:

$$
p_i(t) = \begin{cases}
p_{\text{burst}} = \dfrac{\lambda_{\text{burst}}}{1000} \cdot dt = 0.02
  & B(t) = 1 \quad (\lambda_{\text{burst}} = 200\text{ Hz}) \\[6pt]
p_{\text{bg}}    = \dfrac{\lambda_{\text{bg}}}{1000}    \cdot dt = 0.002
  & B(t) = 0 \quad (\lambda_{\text{bg}}    = 20\text{ Hz})
\end{cases}
$$

The **mean firing rate** in the collective regime is:

$$
\bar{\lambda}_{\text{col}}
  = \phi\,\lambda_{\text{burst}} + (1-\phi)\,\lambda_{\text{bg}}
  = 0.2 \times 200 + 0.8 \times 20
  = 40 + 16 = 56 \text{ Hz}
$$

This is nearly identical to the random regime rate of 60 Hz, **deliberately
matching mean rates so that any difference in the Ising parameters is
attributable to correlation structure, not rate changes**.

#### Key property: induced pairwise correlation

Because all neurons share the same latent state B(t), their spikes are
**conditionally independent** given B but **marginally correlated**. The
covariance between neurons i and j in the collective regime is:

$$
\mathrm{Cov}(s_i, s_j)
  = \mathrm{Var}\bigl(E[s_i \mid B]\bigr)
  = \phi(1-\phi)\,(p_{\text{burst}} - p_{\text{bg}})^2
$$

$$
= 0.2 \times 0.8 \times (0.02 - 0.002)^2
= 0.16 \times 3.24 \times 10^{-4}
\approx 5.2 \times 10^{-5}
$$

While the per-step covariance is small (spikes are rare at 0.1 ms resolution),
**within a 5 ms bin** the burst state is approximately constant, and the
probability that both neurons fired at least once in the bin is substantially
elevated:

$$
P(\text{both fire in 5 ms} \mid \text{burst bin})
  \approx \bigl[1-(1-p_{\text{burst}})^{50}\bigr]^2
  = (0.64)^2 = 0.41
$$

$$
P(\text{both fire in 5 ms} \mid \text{quiet bin})
  \approx \bigl[1-(1-p_{\text{bg}})^{50}\bigr]^2
  = (0.095)^2 = 0.009
$$

This 45-fold ratio between burst-bin and quiet-bin co-firing probability is
what the Ising model detects as elevated pairwise coupling **J**.

---

## 3. Inverse Ising Model

### 3.1 Maximum Entropy Formulation

Given N observed binary neurons with spins σ_i ∈ {−1, +1} (converted from
{0, 1} via σ = 2s − 1), the maximum-entropy distribution that matches the
empirical first and second moments is the **pairwise Ising distribution**:

$$
P(\boldsymbol{\sigma}) = \frac{1}{Z}\exp\!\left(
  \sum_i h_i \sigma_i + \sum_{i < j} J_{ij} \sigma_i \sigma_j
\right)
$$

where:

- **h_i** — local field for neuron i, encoding its mean activity (⟨σ_i⟩)
- **J_{ij}** — pairwise coupling between neurons i and j, encoding their
  co-fluctuations above what the individual rates predict
- **Z** — partition function (normalisation constant)

The **Hamiltonian** (negative log-probability up to a constant) is:

$$
H(\boldsymbol{\sigma}) = -\sum_{i < j} J_{ij} \sigma_i \sigma_j
                         - \sum_i h_i \sigma_i
$$

This can be decomposed into two terms:

$$
H(\boldsymbol{\sigma}) = -\underbrace{\sum_{i < j} J_{ij}\sigma_i\sigma_j}_{J\text{-term}}
                         - \underbrace{\sum_i h_i \sigma_i}_{h\text{-term}}
$$

**J-term** measures the contribution of pairwise correlations to the system's
energy. When neurons are synchronised, many products σ_i σ_j are large and
positive, and if J_{ij} > 0 the J-term becomes large, lowering the Hamiltonian
(more probable state).

**h-term** measures the contribution of individual firing rates. Even
independent neurons with non-zero mean magnetisation ⟨σ_i⟩ contribute through
their local fields h_i.

### 3.2 Spike Data Preprocessing

Raw spikes are recorded at **dt = 0.1 ms** resolution. At this resolution, the
probability of a spike in any single bin is:

$$
P(s_i = 1) = \frac{\lambda_{\text{rand}}}{1000} \times 0.1 = 0.006
\quad \Rightarrow \quad
\langle\sigma_i\rangle = 2 \times 0.006 - 1 = -0.988
$$

With ⟨σ_i⟩ this close to −1, all pairwise correlations approach +1 regardless
of true coupling strength (degenerate statistics). The Ising fit would diverge.

To obtain informative statistics, spikes are binned into **Δ = 5 ms windows**
before fitting:

$$
\sigma_i^{(b)} = \begin{cases}
+1 & \text{if neuron } i \text{ fired at least once in bin } b \\
-1 & \text{otherwise}
\end{cases}
$$

For 60 Hz neurons in a 5 ms bin:

$$
P(\text{spike in 5 ms})
  = 1 - (1 - 0.006)^{50}
  = 1 - 0.994^{50}
  \approx 0.26
\quad\Rightarrow\quad
\langle\sigma\rangle \approx -0.48
$$

This brings the magnetisation into the informative range [−1, +1], making
correlations meaningful for fitting.

The total number of Ising parameters for N neurons is:

$$
N_{\text{params}} = N + \binom{N}{2} = N + \frac{N(N-1)}{2}
$$

For N = 20: 20 + 190 = **210 parameters**.
For N = 10: 10 + 45 = **55 parameters** (default, faster convergence).

### 3.3 Fitting Algorithm — MCH (Mean-field Consistent with Histogram)

The inverse problem is solved using the **Mean-field Consistent Hamiltonian**
(MCH) algorithm implemented in the `coniii` library. MCH iteratively updates
the parameters θ = (h, J) to match the empirical moments:

**Objective:** find θ such that

$$
\langle\sigma_i\rangle_\theta = \langle\sigma_i\rangle_{\text{data}}
\quad \forall i
\qquad \text{and} \qquad
\langle\sigma_i\sigma_j\rangle_\theta = \langle\sigma_i\sigma_j\rangle_{\text{data}}
\quad \forall i < j
$$

**Update rule** at iteration k with learning rate η_k:

$$
h_i \leftarrow h_i + \eta_k\,
  \bigl(\langle\sigma_i\rangle_{\text{data}} - \langle\sigma_i\rangle_{\theta}\bigr)
$$

$$
J_{ij} \leftarrow J_{ij} + \eta_k\,
  \bigl(\langle\sigma_i\sigma_j\rangle_{\text{data}}
       - \langle\sigma_i\sigma_j\rangle_{\theta}\bigr)
$$

Model expectations ⟨·⟩_θ are estimated by **Metropolis–Hastings MCMC** sampling
from P(σ; θ) at each iteration. The learning rate schedule used here is:

$$
\eta_k = \frac{1}{k+1}
\quad (\text{decaying step size for stability})
$$

Convergence is monitored via the L2 norm of the parameter update:

$$
\|\Delta\theta\|_2 = \sqrt{\sum_i (\Delta h_i)^2 + \sum_{i<j}(\Delta J_{ij})^2}
$$

A value below ~0.05 indicates convergence.

---

## 4. Hamiltonian Decomposition Over Time

After fitting on the full spike record (both regimes pooled), a single set of
parameters **(h*, J*)** is obtained. The Hamiltonian is then evaluated at every
time bin b using the fitted parameters and the instantaneous spike pattern:

$$
H(b) = -\sum_{i<j} J^*_{ij}\,\sigma_i^{(b)}\sigma_j^{(b)}
       - \sum_i h^*_i\,\sigma_i^{(b)}
$$

$$
J\text{-term}(b) = \sum_{i<j} J^*_{ij}\,\sigma_i^{(b)}\sigma_j^{(b)}
\qquad
h\text{-term}(b) = \sum_i h^*_i\,\sigma_i^{(b)}
$$

Both terms are smoothed with a **causal Gaussian kernel** (σ ≈ 6 ms) to produce
interpretable time series.

---

## 5. Why the J-Term Jumps During the Collective Regime

This is the central result of the simulation. The argument proceeds in three
steps.

### Step 1: What J encodes

The coupling J_{ij} is positive when neurons i and j co-activate more than
expected from their individual rates alone:

$$
J_{ij} > 0
\;\Longleftrightarrow\;
\langle\sigma_i\sigma_j\rangle > \langle\sigma_i\rangle\langle\sigma_j\rangle
\;\Longleftrightarrow\;
\mathrm{Corr}(\sigma_i,\sigma_j) > 0
$$

In the random regime, neurons fire **independently**, so:

$$
\langle\sigma_i\sigma_j\rangle_{\text{rand}} = \langle\sigma_i\rangle^2
\quad\Rightarrow\quad
J_{ij}^{\text{rand}} \approx 0
$$

In the collective regime, the shared burst/quiet state B(t) creates excess
co-activation. Using the law of total covariance:

$$
\mathrm{Cov}(\sigma_i,\sigma_j)
= \underbrace{E\bigl[\mathrm{Cov}(\sigma_i,\sigma_j\mid B)\bigr]}_{=\;0\;\text{(cond. independent)}}
+ \underbrace{\mathrm{Var}\bigl(E[\sigma_i\mid B]\bigr)}_{\text{variance of the mean}}
$$

$$
= \mathrm{Var}\bigl(E[\sigma_i\mid B]\bigr)
= \phi(1-\phi)\,(\bar\sigma_{\text{burst}} - \bar\sigma_{\text{quiet}})^2
$$

where:

$$
\bar\sigma_{\text{burst}}
  = 2[1-(1-p_{\text{burst}})^{50}]-1
  \approx 2(0.64)-1 = +0.28
  \quad \text{(in 5 ms bins)}
$$

$$
\bar\sigma_{\text{quiet}}
  = 2[1-(1-p_{\text{bg}})^{50}]-1
  \approx 2(0.095)-1 = -0.81
$$

$$
\mathrm{Cov}(\sigma_i,\sigma_j)
= 0.2 \times 0.8 \times (0.28 - (-0.81))^2
= 0.16 \times 1.188
\approx 0.19
$$

This pairwise excess correlation of **ΔC_{ij} ≈ 0.19** drives the Ising
solver to assign positive J values across all pairs. The **J-term** summed over
all pairs therefore rises during collective blocks.

### Step 2: Why the h-term stays roughly flat

The h-term depends on ⟨σ_i⟩ for each neuron. The mean magnetisation in the
collective regime is:

$$
\langle\sigma_i\rangle_{\text{col}}
= \phi\,\bar\sigma_{\text{burst}} + (1-\phi)\,\bar\sigma_{\text{quiet}}
= 0.2 \times 0.28 + 0.8 \times (-0.81)
= 0.056 - 0.648 = -0.59
$$

Compare to the random regime at 60 Hz in 5 ms bins:

$$
\langle\sigma_i\rangle_{\text{rand}}
= 2(0.26)-1 = -0.48
$$

The means differ by only ≈ 0.11, a modest shift that changes h slightly.
Because the Ising fit uses parameters optimised for the pooled dataset, the
h-term tracks the mean magnetisation (rate), which is nearly the same in both
regimes. The h-term contribution therefore does **not** show a large jump.

### Step 3: The J jump is a fingerprint of hidden synchrony

Combining both steps:

$$
\boxed{
\text{J-term}(b) \text{ is large when many pairs } (i,j)
\text{ satisfy } \sigma_i^{(b)}\sigma_j^{(b)} = +1
\text{ simultaneously,}
}
$$

which happens **only during collective burst bins** when the population
co-fires. During random bins the same neurons fire independently and
σ_i^(b) σ_j^(b) averages to ⟨σ⟩² < 0 (most neurons are silent, σ = −1).

Schematically:

```
Random bin:   σ = [−1, +1, −1, −1, +1, −1, −1, +1, −1, −1]   (scattered)
              J-term ≈ Σ J_ij (−1)(−1) ≈ small positive, dominated by h-term

Burst bin:    σ = [+1, +1, +1, +1, −1, +1, +1, +1, −1, +1]   (co-active!)
              J-term ≈ Σ J_ij (+1)(+1)  →  LARGE
              h-term roughly unchanged (similar number of +1 spins expected)
```

The J-term is therefore a **direct readout of the degree of population
synchrony** at each time bin — it jumps whenever many neurons co-activate,
independent of whether the population mean rate changed.

---

## 6. Connection to Purkinje Cell Physiology

In a real reach experiment, the analogue of the "collective regime" is the
movement epoch. Several mechanisms could drive PC synchrony during reach:

1. **Inferior olive (IO) input via climbing fibres**: the IO fires in
   synchronous bursts across adjacent PCs, imposing correlated complex-spike
   responses. These brief depolarisations create co-active epochs in the simple
   spike train.

2. **Common parallel-fibre (granule cell) input**: during movement, granule
   cells activated by mossy fibres carrying proprioceptive signals converge on
   overlapping PC dendrites, creating correlated excitatory drive.

3. **Basket / stellate cell lateral inhibition**: molecular layer interneurons
   couple neighbouring PCs via inhibitory synapses. A synchronised inhibitory
   wave followed by post-inhibitory rebound can synchronise PC firing.

In all three cases the result is the same: **epochs of co-activation and
co-pause** across the PC population, exactly what the burst/quiet Markov chain
models. The Ising J-term rises because these shared states create excess
pairwise co-activation that cannot be explained by the individual firing rates
alone.

---

## 7. Parameter Reference

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--ne` | 800 (100 for PC) | Number of principal neurons |
| `--ni` | 200 (25 for PC) | Number of interneurons |
| `--rand_rate` | 60 Hz | Poisson rate in random (pre-reach) regime |
| `--burst_rate` | 200 Hz | Firing rate during population burst events |
| `--bg_rate` | 20 Hz | Background rate between bursts (collective) |
| `--p_burst_on` | 0.005 | Quiet → burst transition prob/step (~20 ms quiet) |
| `--p_burst_off` | 0.02 | Burst → quiet transition prob/step (~5 ms burst) |
| `--seg_len` | 2000 | Steps per regime block (200 ms at dt = 0.1 ms) |
| `--total_steps` | 5000 | Total simulation duration (500 ms) |
| `--bin_ms` | 5.0 | Spike bin width for Ising fitting (ms) |
| `--n_ising` | 20 | Neurons sampled for Ising fit |
| `--sample_size` | 50000 | MCMC samples per MCH iteration |
| `--max_iter` | 100 | Maximum MCH iterations |

---

## 8. Summary

The simulation demonstrates that **the J-term of the Ising Hamiltonian is a
sensitive detector of population synchrony** that is blind to mean-rate changes.
A two-state Markov burst/quiet model — with parameters matched to approximate
Purkinje cell physiology — produces a statistically robust J jump during the
collective blocks at nearly identical mean firing rates.

This provides a concrete mechanistic hypothesis for the J jump observed in real
PC data during a mouse arm reach: **Purkinje cells transiently synchronise
during movement, creating co-active epochs that the Ising pairwise coupling J
must absorb to explain the data**. The h-term, which tracks individual rates,
remains relatively stable, while J carries the entire signal of population-level
coordination.
