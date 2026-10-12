# Purkinje Cell Population Simulation and Inverse Ising Analysis

## Overview

This document describes the mathematical framework behind the Purkinje cell (PC)
population simulation, the inverse Ising model fitted to the spike data, and a
mechanistic explanation of why the pairwise coupling term **J jumps during the
collective (reach) regime** even when mean firing rates are nearly identical
across regimes. Each mathematical statement is followed immediately by the
corresponding code from `simulate_neural_regimes.py`.

---

## 1. Biological Motivation

Purkinje cells are the sole output neurons of the cerebellar cortex. During a
mouse arm reach, multi-unit spike data show a detectable change in the
**pairwise coupling** extracted by maximum-entropy (Ising) models, while
single-unit firing rates change only modestly. This simulation isolates the
mechanism: **population synchrony alone — without any rate change — is
sufficient to produce a J jump**.

Two behavioural states are modelled:

| State | Biology | Simulation |
|-------|---------|------------|
| **Random / pre-reach** | PC simple spikes at spontaneous rate, uncorrelated | Independent Poisson at λ_rand ≈ 60 Hz |
| **Collective / during reach** | PC populations co-activate and co-pause in synchrony | Shared burst/quiet Markov state at mean ≈ 56 Hz |

---

## 2. Neural Population Model

### 2.1 Populations and Time Axis

Two populations are simulated:

| Population | Label | Default size | Biology |
|------------|-------|-------------|---------|
| Principal | Purkinje cells | N_e = 100 | GABAergic output cells |
| Interneurons | Basket / Stellate | N_i = 25 | Molecular layer inhibitory cells |

Time is discretised at **dt = 0.1 ms** for a total of **T_total = 500 ms
(5 000 steps)**. Each step t ∈ {0, 1, …, N_t − 1} produces a binary spike
vector for each population.

```python
dt          = 0.1           # ms per step
TOTAL_STEPS = args.total_steps   # default 5 000
T           = TOTAL_STEPS * dt   # total duration in ms  (500 ms)
time_ax     = np.arange(0, T, dt)
Nt          = len(time_ax)

# Network dimensions
Ne = args.ne   # default 800 (use 100 for Purkinje-like)
Ni = args.ni   # default 200 (use 25  for Basket/Stellate)
```

---

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

The first block is always **random** (pre-reach baseline), and the second block
is **collective** (reach). The code builds the `mode` array by setting every
second L-length segment to 1:

```python
SEG  = args.seg_len          # default 2 000 steps = 200 ms
mode = np.zeros(Nt, dtype=int)
for blk in range(SEG, Nt, 2 * SEG):
    mode[blk : blk + SEG] = 1   # collective blocks
```

---

### 2.3 Per-Step Firing Probabilities

Each of the three firing rates (random regime, collective-burst, collective-quiet)
is converted from Hz to a **per-step Bernoulli probability**:

$$
p = \frac{\lambda \,[\text{Hz}]}{1000} \times dt \,[\text{ms}]
$$

For example, with λ_rand = 60 Hz and dt = 0.1 ms:

$$
p_{\text{rand}} = \frac{60}{1000} \times 0.1 = 6 \times 10^{-3}
$$

```python
RAND_P  = args.rand_rate  / 1000 * dt   # random regime      (default 60 Hz  → 0.006)
BURST_P = args.burst_rate / 1000 * dt   # collective burst   (default 200 Hz → 0.020)
BG_P    = args.bg_rate    / 1000 * dt   # collective quiet   (default 20 Hz  → 0.002)
```

---

### 2.4 Collective Regime — Shared Burst / Quiet Markov State

During collective blocks a **shared population state** B(t) ∈ {0, 1} evolves
as a two-state discrete-time Markov chain. At each step:

$$
P\bigl(B(t+1) = 1 \mid B(t) = 0\bigr) = p_{\text{on}}  = 0.005
\qquad \text{(quiet} \to \text{burst)}
$$

$$
P\bigl(B(t+1) = 0 \mid B(t) = 1\bigr) = p_{\text{off}} = 0.02
\qquad \text{(burst} \to \text{quiet)}
$$

```python
P_BURST_ON  = args.p_burst_on    # default 0.005   quiet → burst
P_BURST_OFF = args.p_burst_off   # default 0.02    burst → quiet
```

The **stationary burst fraction** (fraction of collective-block time spent in
the burst state) is derived from the balance of the Markov chain:

$$
\phi = \frac{p_{\text{on}}}{p_{\text{on}} + p_{\text{off}}}
    = \frac{0.005}{0.005 + 0.02} = 0.20
$$

Mean sojourn times follow geometric distributions (number of steps until
transition × dt):

$$
\tau_{\text{burst}} = \frac{dt}{p_{\text{off}}} = \frac{0.1}{0.02} = 5 \text{ ms}
\qquad
\tau_{\text{quiet}} = \frac{dt}{p_{\text{on}}}  = \frac{0.1}{0.005} = 20 \text{ ms}
$$

```python
burst_frac  = P_BURST_ON / (P_BURST_ON + P_BURST_OFF)   # = 0.20
mean_col_hz = burst_frac * args.burst_rate + (1 - burst_frac) * args.bg_rate

# Logged at startup:
# Burst fraction ≈ 0.20  →  mean collective rate ≈ 56 Hz
# Mean burst duration ≈ 5 ms  |  mean quiet duration ≈ 20 ms
```

---

### 2.5 The Simulation Loop

At every time step the regime indicator `mode[i]` determines which branch
runs. In the collective branch the Markov state is updated first, then a
per-neuron Bernoulli draw is made at the state-dependent rate. In the random
branch every neuron draws independently at `RAND_P`.

**Conditional firing probability** given the current state B:

$$
p_i(t \mid B) = \begin{cases}
p_{\text{burst}} = 0.020 & B(t) = 1 \quad (200 \text{ Hz burst}) \\
p_{\text{bg}}    = 0.002 & B(t) = 0 \quad (20 \text{ Hz quiet})
\end{cases}
\qquad s_i(t) \sim \mathrm{Bernoulli}(p_i)
$$

Because all N_e neurons share **the same B(t)** but draw **independently
given B**, their spikes are marginally correlated — the source of the J jump.

**Mean firing rate** in the collective regime:

$$
\bar{\lambda}_{\text{col}}
  = \phi\,\lambda_{\text{burst}} + (1-\phi)\,\lambda_{\text{bg}}
  = 0.2 \times 200 + 0.8 \times 20 = 56 \text{ Hz}
  \;\approx\; \lambda_{\text{rand}} = 60 \text{ Hz}
$$

This deliberate rate-matching ensures any difference in Ising parameters is
due to **correlation structure, not rate changes**.

```python
Se = np.zeros((Ne, Nt), dtype=np.float32)
Si = np.zeros((Ni, Nt), dtype=np.float32)
pop_state_arr = np.zeros(Nt, dtype=np.int8)   # records B(t): 0=quiet, 1=burst

pop_state = 0   # B(t); only advances during collective blocks

for i in range(Nt - 1):
    if mode[i] == 1:                           # ── COLLECTIVE block ──
        # Markov transition: B(t) → B(t+1)
        if pop_state == 0:
            pop_state = 1 if (np.random.rand() < P_BURST_ON) else 0
        else:
            pop_state = 0 if (np.random.rand() < P_BURST_OFF) else 1
        pop_state_arr[i] = pop_state

        # State-dependent firing rate
        rate_e = BURST_P if pop_state == 1 else BG_P
        rate_i = BURST_P if pop_state == 1 else BG_P

    else:                                      # ── RANDOM block ──
        pop_state = 0      # reset: no burst bleeds into random blocks
        pop_state_arr[i] = 0
        rate_e = RAND_P
        rate_i = RAND_P * 0.5   # interneurons slightly quieter at baseline

    # Independent Bernoulli draw for each neuron
    idx_e = np.where(np.random.rand(Ne) < rate_e)[0]
    idx_i = np.where(np.random.rand(Ni) < rate_i)[0]
    Se[idx_e, i + 1] = 1 / dt
    Si[idx_i, i + 1] = 1 / dt
```

---

### 2.6 Burst-State Tracking for Visualisation

After the loop, `pop_state_arr` records B(t) at every step. The code scans
this array to find the start and end times of each burst epoch, which are then
shaded on the raster plot as dark-red vertical bands:

```python
burst_spans, in_burst = [], False
for i in range(Nt):
    if pop_state_arr[i] == 1 and not in_burst:
        bs_t, in_burst = time_ax[i], True          # burst starts
    elif pop_state_arr[i] == 0 and in_burst:
        burst_spans.append((bs_t, time_ax[i]))     # burst ends
        in_burst = False
if in_burst:
    burst_spans.append((bs_t, time_ax[-1]))        # burst at end of record
```

---

### 2.7 Induced Pairwise Correlation (why J will jump)

Because neurons are **conditionally independent** given B(t) but share the
same hidden state, the **law of total covariance** gives:

$$
\mathrm{Cov}(s_i, s_j)
= \underbrace{E\bigl[\mathrm{Cov}(s_i,s_j\mid B)\bigr]}_{=\;0}
+ \underbrace{\mathrm{Var}\bigl(E[s_i\mid B]\bigr)}_{\phi(1-\phi)(p_{\text{burst}}-p_{\text{bg}})^2}
$$

$$
= \phi(1-\phi)\,(p_{\text{burst}} - p_{\text{bg}})^2
= 0.2 \times 0.8 \times (0.020 - 0.002)^2
\approx 5.2 \times 10^{-5}
$$

This per-step covariance is small (spikes are rare at 0.1 ms resolution), but
after binning into **5 ms windows** the signal becomes clear. If the bin
contains a burst epoch, the probability that **both** neurons i and j fired is:

$$
P(\text{co-fire in 5 ms} \mid \text{burst bin})
  \approx \bigl[1-(1-p_{\text{burst}})^{50}\bigr]^2 = (0.64)^2 = 0.41
$$

$$
P(\text{co-fire in 5 ms} \mid \text{quiet bin})
  \approx \bigl[1-(1-p_{\text{bg}})^{50}\bigr]^2 = (0.095)^2 = 0.009
$$

A **45-fold** difference in pairwise co-firing between burst and quiet bins is
the statistical signal the Ising model encodes as a positive J coupling.

---

## 3. Inverse Ising Model

### 3.1 Maximum Entropy Formulation

Given N neurons with spins σ_i ∈ {−1, +1} (converted from {0,1} via σ = 2s−1),
the maximum-entropy distribution matching empirical first and second moments is
the **pairwise Ising distribution**:

$$
P(\boldsymbol{\sigma}) = \frac{1}{Z}\exp\!\left(
  \sum_i h_i \sigma_i + \sum_{i < j} J_{ij} \sigma_i \sigma_j
\right)
$$

The **Hamiltonian** decomposes into two interpretable terms:

$$
H(\boldsymbol{\sigma})
  = -\underbrace{\sum_{i<j} J_{ij}\sigma_i\sigma_j}_{J\text{-term (synchrony)}}
    -\underbrace{\sum_i h_i\sigma_i}_{h\text{-term (rate)}}
$$

- **h_i** encodes mean activity of neuron i (rate)
- **J_{ij}** encodes excess pairwise co-activation above what individual rates predict (synchrony)

---

### 3.2 Spike Data Preprocessing — Binning to 5 ms

Raw 0.1 ms spikes give ⟨σ_i⟩ ≈ −0.99 (neurons are almost never active per step).
All correlations then trivially approach +1, making the fit degenerate. Binning
into 5 ms windows brings ⟨σ⟩ into the informative range:

$$
P(\text{spike in 5 ms}) = 1-(1-0.006)^{50} \approx 0.26
\quad\Rightarrow\quad \langle\sigma\rangle = 2(0.26)-1 = -0.48
$$

```python
BIN_STEPS   = max(1, round(args.bin_ms / dt))    # steps per bin  (default: 50 = 5 ms)
n_bins      = Nt // BIN_STEPS                    # number of Ising time bins
time_binned = np.arange(n_bins) * args.bin_ms    # ms axis for energy plots

# Subsample N_ISING principal neurons
N_ISING = args.n_ising                           # default 20  (→ 210 parameters)
neu_idx = np.random.choice(Ne, N_ISING, replace=False)

# Binary: did neuron n fire at least once in bin b?
Se_sub    = Se[neu_idx, : n_bins * BIN_STEPS]             # (N_ISING, n_bins*BIN_STEPS)
Se_binned = Se_sub.reshape(N_ISING, n_bins, BIN_STEPS)    # (N_ISING, n_bins, BIN_STEPS)
spikes_01 = (Se_binned.max(axis=2) > 0).astype(np.int64).T  # (n_bins, N_ISING)
spikes_pm = (2 * spikes_01 - 1).astype(np.int64)            # σ ∈ {-1, +1}
```

The total number of Ising parameters for N neurons is:

$$
N_{\text{params}} = N + \binom{N}{2} = N + \frac{N(N-1)}{2}
\quad
\begin{cases} N=20 \to 210 \\ N=10 \to 55 \end{cases}
$$

---

### 3.3 Fitting — MCH Algorithm with Decaying Learning Rate

The MCH solver iteratively adjusts θ = (h, J) until the model moments match
the data moments:

$$
h_i \leftarrow h_i + \eta_k \bigl(\langle\sigma_i\rangle_{\text{data}} - \langle\sigma_i\rangle_\theta\bigr)
\qquad
J_{ij} \leftarrow J_{ij} + \eta_k \bigl(\langle\sigma_i\sigma_j\rangle_{\text{data}} - \langle\sigma_i\sigma_j\rangle_\theta\bigr)
$$

with decaying learning rate η_k = 1/(k+1). Model expectations are estimated
by Metropolis MCMC at each iteration.

```python
from coniii import MCH

solver = MCH(
    spikes_pm,
    sample_size = args.sample_size,         # default 50 000 MCMC samples
    rng         = np.random.RandomState(args.seed),
    n_cpus      = args.n_cpus,
    sampler_kw  = {'boost': True},
)

def _learn(i):
    return {'maxdlamda': 1, 'eta': 1 / (i + 1)}   # η_k = 1/(k+1)

multipliers = solver.solve(
    maxiter  = args.max_iter,                       # default 100 iterations
    n_iters  = max(N_ISING * 100, 2000),            # MCMC steps per iteration
    burn_in  = max(N_ISING * 50,  1000),
    iprint   = 'detailed',
    custom_convergence_f = _learn,
)
```

Convergence is monitored via the L2 norm of the parameter update step:

$$
\|\Delta\theta\|_2 = \sqrt{\sum_i (\Delta h_i)^2 + \sum_{i<j}(\Delta J_{ij})^2}
\quad \text{(target < 0.05)}
$$

---

## 4. Hamiltonian Decomposition Over Time

After fitting, the single set of parameters (h*, J*) is evaluated at every
Ising bin b to produce time-resolved energy, J-term, and h-term traces:

$$
H(b) = -\sum_{i<j} J^*_{ij}\,\sigma_i^{(b)}\sigma_j^{(b)}
       - \sum_i h^*_i\,\sigma_i^{(b)}
$$

$$
J\text{-term}(b) = \sum_{i<j} J^*_{ij}\,\sigma_i^{(b)}\sigma_j^{(b)}
\qquad
h\text{-term}(b) = \sum_i h^*_i\,\sigma_i^{(b)}
$$

```python
def _calc_e_with_terms_np(s, params):
    """H(σ) = -J_term - h_term.  Returns (energy, J_term, h_term), shape (T,)."""
    N   = s.shape[1]
    h_p = params[:N]      # local fields  h_i
    J_p = params[N:]      # couplings     J_ij  (upper-triangle, row-major)
    h_t = s @ h_p         # h-term:  Σ_i h_i σ_i  for each bin  → shape (T,)
    J_t = np.zeros(s.shape[0])
    k = 0
    for ii in range(N - 1):
        for jj in range(ii + 1, N):
            J_t += J_p[k] * s[:, ii] * s[:, jj]   # J-term: Σ_{i<j} J_ij σ_i σ_j
            k   += 1
    return -J_t - h_t, J_t, h_t

energies_t, J_terms_t, h_terms_t = calc_e_with_terms(
    spikes_pm, np.asarray(multipliers, dtype=np.float64))
```

Both series are smoothed with a **causal Gaussian kernel** (σ ≈ 6 ms in bin
units) so the plots reflect trends rather than per-bin noise:

```python
sigma_b  = max(1, round(6 / args.bin_ms))       # kernel width in bin units
s_b      = np.arange(-3*sigma_b, 3*sigma_b + 1)
kern_b   = np.exp(-(s_b**2) / (2 * sigma_b**2))
kern_b[s_b < 0] = 0                             # causal: zero weight for future
kern_b  /= kern_b.sum()
J_s      = np.convolve(kern_b, J_terms_t, 'same')
h_s      = np.convolve(kern_b, h_terms_t, 'same')
```

---

## 5. Why the J-Term Jumps During the Collective Regime

### Step 1 — J encodes excess pairwise co-activation

The coupling J_{ij} is positive when neurons i and j co-activate more than
their individual rates predict:

$$
J_{ij} > 0
\;\Longleftrightarrow\;
\langle\sigma_i\sigma_j\rangle > \langle\sigma_i\rangle\langle\sigma_j\rangle
$$

In the random regime, neurons are i.i.d., so Cov = 0 and J ≈ 0.

In the collective regime the **law of total covariance** shows that the shared
state B creates nonzero covariance even though neurons are conditionally
independent given B:

$$
\mathrm{Cov}(\sigma_i,\sigma_j)
= \underbrace{E[\mathrm{Cov}(\sigma_i,\sigma_j\mid B)]}_{0}
+ \mathrm{Var}(E[\sigma_i\mid B])
= \phi(1-\phi)(\bar\sigma_{\text{burst}} - \bar\sigma_{\text{quiet}})^2 \approx 0.19
$$

where (in 5 ms bins):

$$
\bar\sigma_{\text{burst}} = 2[1-(1-0.020)^{50}]-1 \approx +0.28
\qquad
\bar\sigma_{\text{quiet}} = 2[1-(1-0.002)^{50}]-1 \approx -0.81
$$

### Step 2 — h-term stays flat because mean rates barely differ

$$
\langle\sigma\rangle_{\text{col}} = 0.2(+0.28) + 0.8(-0.81) = -0.59
\qquad
\langle\sigma\rangle_{\text{rand}} = 2(0.26)-1 = -0.48
$$

Δ⟨σ⟩ ≈ 0.11 — small enough that h-term moves only slightly across regimes.

### Step 3 — J-term is a direct readout of synchrony

```
Random bin:   σ = [−1, +1, −1, −1, +1, −1, −1, +1, −1, −1]   (scattered)
              J-term = Σ J_ij σ_i σ_j  → many (−1)(−1) = +1 pairs, but so does h-term;
              net effect small, h-term dominates.

Burst bin:    σ = [+1, +1, +1, +1, −1, +1, +1, +1, −1, +1]   (co-active!)
              J-term = Σ J_ij (+1)(+1)  →  LARGE positive jump
              h-term unchanged (similar count of +1 spins)
```

$$
\boxed{
\text{J-term rises whenever many neuron pairs satisfy } \sigma_i\sigma_j = +1
\text{ simultaneously — i.e., during population bursts.}
}
$$

---

## 6. Connection to Purkinje Cell Physiology

The burst/quiet Markov chain mimics three known mechanisms that synchronise
PC populations during a reach:

1. **Inferior olive → climbing fibres**: IO fires synchronous bursts across
   adjacent PCs, imposing correlated complex-spike events that briefly elevate
   simple-spike rates together.

2. **Common parallel-fibre (granule cell) input**: mossy-fibre proprioceptive
   signals activate overlapping granule-cell populations whose parallel fibres
   converge on neighbouring PC dendrites, creating a shared excitatory wave.

3. **Basket / stellate cell lateral inhibition**: a synchronised inhibitory
   volley followed by post-inhibitory rebound co-activates a PC ensemble.

In all three cases the result is the same: **epochs of population co-activation
and co-pause** — exactly what the Markov burst/quiet model produces — and the
Ising J-term rises to capture these correlated epochs.

---

## 7. Parameter Reference

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--ne` | 800 (100 for PC) | Number of principal neurons |
| `--ni` | 200 (25 for PC) | Number of interneurons |
| `--rand_rate` | 60 Hz | Poisson rate in random / pre-reach regime |
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
sensitive detector of population synchrony** that is blind to mean-rate
changes. The two-state Markov burst/quiet model — parameters matched to
Purkinje cell physiology — produces a robust J jump during collective blocks at
nearly identical mean firing rates. Each mathematical claim in this document is
directly traceable to a line of code in `simulate_neural_regimes.py`.

This provides a concrete mechanistic hypothesis for the J jump in real PC data
during a mouse arm reach: **Purkinje cells transiently synchronise during
movement, and the Ising pairwise coupling J must rise to account for the
co-active epochs that individual firing rates cannot explain**.
