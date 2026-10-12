#!/usr/bin/env python3
"""
Simulates a spiking neural network that alternates between:
  - Random / independent Poisson regime (recurrent connections disabled)
  - Recurrent / collective regime (full EI network dynamics)

Then fits an inverse Ising model on a subsampled set of excitatory neurons,
computes the Hamiltonian decomposition H(t) = -J_term(t) - h_term(t) per
time step, and saves visualisations.

Schedule: 200-step blocks, alternating Random → Recurrent → Random → …
Total: 5 000 time steps (500 ms at dt = 0.1 ms)

Outputs (notes/neural_simulation/ by default):
  spike_activity_overview.png   raster + rate + energy + J/h terms
  spike_raster_detailed.png     separate E/I rasters + rate
  block_firing_rates.png        mean rate per 200-step block
  model_quality_summary.png     pairwise/triplet correlations, P(K), J-dist
  correlation_order_*.png       k-order correlations (from plot_model_quality)
  summary.txt                   numeric summary

Usage:
  python simulate_neural_regimes.py [--output_dir DIR] [--seed N]
                                    [--total_steps N] [--seg_len N]
                                    [--n_ising N] [--n_cpus N]
                                    [--max_iter N] [--sample_size N]
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import time as tm

# ── Local modules (model.py / visualization.py sit next to this script) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── CLI arguments ──────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description='EI network simulation + inverse Ising analysis')
    p.add_argument('--output_dir',  type=str,   default=None,
                   help='Output directory (default: <script_dir>/notes/neural_simulation)')
    p.add_argument('--seed',        type=int,   default=42,
                   help='NumPy random seed (default: 42)')
    p.add_argument('--total_steps', type=int,   default=5_000,
                   help='Total simulation time steps (default: 5000 = 500 ms at dt=0.1 ms)')
    p.add_argument('--seg_len',     type=int,   default=2000,
                   help='Steps per random/recurrent block (default: 2000 = 200 ms)')
    p.add_argument('--bin_ms',      type=float, default=0.1,
                   help='Spike-bin width in ms before Ising fitting (default: 0.1 = no '
                        'binning, one raw time step per Ising sample). Set to e.g. 5.0 '
                        'to coarsen bins and move ⟨σ⟩ toward 0 for easier convergence.')
    p.add_argument('--ne',          type=int,   default=800,
                   help='Number of principal (E) neurons (default: 800). '
                        'For Purkinje-like: use 100–200.')
    p.add_argument('--ni',          type=int,   default=200,
                   help='Number of interneurons (I) (default: 200). '
                        'For Purkinje-like: use 25–50 (basket/stellate cells).')
    p.add_argument('--nx',          type=int,   default=800,
                   help='Number of external Poisson neurons (default: 800). '
                        'For Purkinje-like: parallel fibre / granule cell input.')
    p.add_argument('--label_e',     type=str,   default='E neurons',
                   help='Display label for the principal population '
                        '(default: "E neurons"; use "Purkinje cells" for cerebellar sims)')
    p.add_argument('--label_i',     type=str,   default='I neurons',
                   help='Display label for the interneuron population '
                        '(default: "I neurons"; use "Basket/Stellate" for cerebellar sims)')
    p.add_argument('--rand_rate',   type=float, default=60.0,
                   help='Independent Poisson firing rate (Hz) in the random/pre-reach regime '
                        '(default: 60 Hz ≈ Purkinje spontaneous rate). '
                        'This regime has no pairwise correlations → h-term dominates.')
    p.add_argument('--burst_rate',  type=float, default=200.0,
                   help='Population firing rate (Hz) during synchronized bursts in collective '
                        'regime (default: 200 Hz). All neurons co-fire → J-term dominates.')
    p.add_argument('--bg_rate',     type=float, default=20.0,
                   help='Background firing rate (Hz) between bursts in collective regime '
                        '(default: 20 Hz). '
                        'Mean collective rate ≈ burst_frac*burst_rate + (1-burst_frac)*bg_rate.')
    p.add_argument('--p_burst_on',  type=float, default=0.005,
                   help='Prob/step of quiet→burst transition in collective mode '
                        '(default: 0.005 → mean quiet duration ≈ 20 ms). '
                        'Burst fraction = p_burst_on/(p_burst_on+p_burst_off) ≈ 0.20.')
    p.add_argument('--p_burst_off', type=float, default=0.02,
                   help='Prob/step of burst→quiet transition in collective mode '
                        '(default: 0.02 → mean burst duration ≈ 5 ms).')
    p.add_argument('--n_ising',     type=int,   default=20,
                   help='Neurons used for Ising fit (default: 20; fewer = faster convergence)')
    p.add_argument('--rx',          type=float, default=3.5,
                   help='[legacy, ignored] External drive Hz for EIF network.')
    p.add_argument('--n_cpus',      type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
                   help='CPUs for Ising solver (default: SLURM_CPUS_PER_TASK or 4)')
    p.add_argument('--max_iter',    type=int,   default=100,
                   help='MCH solver max iterations (default: 100)')
    p.add_argument('--sample_size', type=int,   default=50_000,
                   help='MCH / Metropolis sample size (default: 50000)')
    return p.parse_args()

args = _parse_args()

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = (os.path.abspath(args.output_dir) if args.output_dir
           else os.path.join(_HERE, 'notes', 'neural_simulation'))
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'font.size':         12,
    'lines.linewidth':   2.2,
    'legend.framealpha': 0.9,
})

np.random.seed(args.seed)

# ── Population labels (used throughout plots) ──────────────────────────────────
LBL_E = args.label_e   # e.g. "E neurons" or "Purkinje cells"
LBL_I = args.label_i   # e.g. "I neurons" or "Basket/Stellate"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Population-synchrony simulation
#
#  RANDOM regime  (pre-reach baseline):
#    Every neuron fires as independent Poisson at RAND_RATE Hz.
#    No pairwise correlations → Ising h-term dominates.
#
#  COLLECTIVE regime  (during reach):
#    A shared population state (BURST / QUIET) drives co-firing.
#    State evolves as a 2-state Markov chain each dt step:
#      quiet → burst  with probability p_burst_on   (default ~20 ms mean quiet)
#      burst → quiet  with probability p_burst_off  (default ~5 ms mean burst)
#    During BURST  : all neurons fire at BURST_RATE Hz  (synchronized)
#    During QUIET  : neurons fire at BG_RATE Hz         (independent)
#    The shared state creates strong pairwise correlations → J-term dominates.
#    Mean rate ≈ burst_frac * BURST_RATE + (1-burst_frac) * BG_RATE  ≈ RAND_RATE
#    (same mean rate as random, different correlation structure — isolates J vs h)
# ══════════════════════════════════════════════════════════════════════════════

dt          = 0.1
TOTAL_STEPS = args.total_steps
T           = TOTAL_STEPS * dt
time_ax     = np.arange(0, T, dt)
Nt          = len(time_ax)

# Regime schedule: mode==0 → random, mode==1 → collective
SEG  = args.seg_len
mode = np.zeros(Nt, dtype=int)
for blk in range(SEG, Nt, 2 * SEG):
    mode[blk : blk + SEG] = 1

# Network dimensions (Ne = principal pop, Ni = interneuron pop)
Ne = args.ne;  Ni = args.ni

# Per-step firing probabilities
RAND_P      = args.rand_rate  / 1000 * dt   # random regime
BURST_P     = args.burst_rate / 1000 * dt   # collective burst state
BG_P        = args.bg_rate    / 1000 * dt   # collective quiet state
P_BURST_ON  = args.p_burst_on               # quiet → burst transition prob
P_BURST_OFF = args.p_burst_off              # burst → quiet transition prob

burst_frac  = P_BURST_ON / (P_BURST_ON + P_BURST_OFF)
mean_col_hz = burst_frac * args.burst_rate + (1 - burst_frac) * args.bg_rate
print(f'Simulation design:')
print(f'  Random   regime: {args.rand_rate:.0f} Hz independent Poisson')
print(f'  Collective regime: {args.burst_rate:.0f} Hz burst  /  {args.bg_rate:.0f} Hz quiet')
print(f'  Burst fraction ≈ {burst_frac:.2f}  →  mean collective rate ≈ {mean_col_hz:.0f} Hz')
print(f'  Mean burst duration ≈ {1/P_BURST_OFF*dt:.0f} ms  |  '
      f'mean quiet duration ≈ {1/P_BURST_ON*dt:.0f} ms')

Se = np.zeros((Ne, Nt), dtype=np.float32)
Si = np.zeros((Ni, Nt), dtype=np.float32)
pop_state_arr = np.zeros(Nt, dtype=np.int8)  # 0=quiet/random, 1=burst

pop_state = 0  # population burst state (only active in collective blocks)

print('\nRunning population-synchrony simulation …')
t0 = tm()

for i in range(Nt - 1):
    if mode[i] == 1:
        # Update shared population state (Markov chain)
        if pop_state == 0:
            pop_state = 1 if (np.random.rand() < P_BURST_ON) else 0
        else:
            pop_state = 0 if (np.random.rand() < P_BURST_OFF) else 1
        pop_state_arr[i] = pop_state
        rate_e = BURST_P if pop_state == 1 else BG_P
        rate_i = BURST_P if pop_state == 1 else BG_P
    else:
        # Enter random block → reset state so no burst bleeds in
        pop_state = 0
        pop_state_arr[i] = 0
        rate_e = RAND_P
        rate_i = RAND_P * 0.5   # interneurons slightly quieter at baseline

    idx_e = np.where(np.random.rand(Ne) < rate_e)[0]
    idx_i = np.where(np.random.rand(Ni) < rate_i)[0]
    Se[idx_e, i + 1] = 1 / dt
    Si[idx_i, i + 1] = 1 / dt

print(f'Simulation finished in {tm()-t0:.1f} s')
n_bursts = int(np.diff(np.concatenate(([0], pop_state_arr))).clip(0).sum())
print(f'  Burst events fired: {n_bursts}')

# ── Basic post-processing ──────────────────────────────────────────────────────
SeIdx, SeTimes_i = np.nonzero(Se);  SeTimes = SeTimes_i * dt
SiIdx, SiTimes_i = np.nonzero(Si);  SiTimes = SiTimes_i * dt

re_raw = np.mean(Se, axis=0)
ri_raw = np.mean(Si, axis=0)

# Causal Gaussian smoothing kernel
sigma  = 6
s_arr  = np.arange(-3*sigma, 3*sigma, dt)
kernel = np.exp(-(s_arr**2) / (2*sigma**2))
kernel[s_arr < 0] = 0
kernel /= kernel.sum() * dt
re = np.convolve(kernel, re_raw, 'same') * dt
ri = np.convolve(kernel, ri_raw, 'same') * dt

# Recurrent spans for axis shading
rec_spans, in_rec = [], False
for i in range(Nt):
    if mode[i] == 1 and not in_rec:
        span_s, in_rec = time_ax[i], True
    elif mode[i] == 0 and in_rec:
        rec_spans.append((span_s, time_ax[i]));  in_rec = False
if in_rec:
    rec_spans.append((span_s, time_ax[-1]))

REC_COLOR   = '#FF8C00';  REC_ALPHA   = 0.20
BURST_COLOR = '#c0392b';  BURST_ALPHA = 0.35

# Burst spans (within collective blocks) for raster overlay
burst_spans, in_burst = [], False
for i in range(Nt):
    if pop_state_arr[i] == 1 and not in_burst:
        bs_t, in_burst = time_ax[i], True
    elif pop_state_arr[i] == 0 and in_burst:
        burst_spans.append((bs_t, time_ax[i]));  in_burst = False
if in_burst:
    burst_spans.append((bs_t, time_ax[-1]))

def shade(ax):
    for (ts, te) in rec_spans:
        ax.axvspan(ts, te, alpha=REC_ALPHA, color=REC_COLOR, zorder=0)

def shade_bursts(ax):
    """Highlight individual synchronized burst events within collective blocks."""
    for (ts, te) in burst_spans:
        ax.axvspan(ts, te, alpha=BURST_ALPHA, color=BURST_COLOR, zorder=1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Inverse Ising model fit
# ══════════════════════════════════════════════════════════════════════════════

# ── Coarsen spikes to BIN_MS windows before Ising fitting ─────────────────────
#
# Raw dt=0.1 ms bins give ⟨σ⟩ ≈ −0.998 even for 60 Hz neurons (P≈0.006/bin).
# All pairwise correlations then ≈ +1 regardless of real coupling, which
# causes the coniii warning "correlations close to one" and diverging J params.
#
# With 5 ms bins:
#   Random   (10 Hz): P(spike in bin) ≈ 0.049  →  ⟨σ⟩ ≈ −0.90
#   Recurrent(59 Hz): P(spike in bin) ≈ 0.256  →  ⟨σ⟩ ≈ −0.49  ✓
# Correlations become informative and the fit converges properly.
BIN_STEPS  = max(1, round(args.bin_ms / dt))   # raw steps per Ising bin
n_bins     = Nt // BIN_STEPS                   # number of Ising time bins
time_binned = np.arange(n_bins) * args.bin_ms  # ms, for energy/term plots

# Subsample N_ISING excitatory neurons (tractable: 20 neurons → 210 params)
N_ISING = args.n_ising
neu_idx = np.random.choice(Ne, N_ISING, replace=False)

# Did neuron n fire at all during bin b?  Shape: (n_bins, N_ISING)
Se_sub     = Se[neu_idx, : n_bins * BIN_STEPS]              # (N_ISING, n_bins*BIN_STEPS)
Se_binned  = Se_sub.reshape(N_ISING, n_bins, BIN_STEPS)     # (N_ISING, n_bins, BIN_STEPS)
spikes_01  = (Se_binned.max(axis=2) > 0).astype(np.int64).T # (n_bins, N_ISING)
spikes_pm  = (2 * spikes_01 - 1).astype(np.int64)           # {-1, +1}

# Binned regime label for each Ising bin (majority-vote)
mode_binned = mode[: n_bins * BIN_STEPS].reshape(n_bins, BIN_STEPS).max(axis=1)

# Print diagnostics so we can verify statistics are sensible
mean_spins = spikes_pm.mean(axis=0)
print(f'\nIsing data: {n_bins} bins  ({BIN_STEPS} steps = {args.bin_ms:.1f} ms each)')
print(f'  ⟨σ⟩ range: [{mean_spins.min():.3f}, {mean_spins.max():.3f}]  '
      f'(target: not too close to ±1)')
rand_bins = (mode_binned == 0).sum()
rec_bins  = (mode_binned == 1).sum()
print(f'  Regime split: {rand_bins} random bins  |  {rec_bins} recurrent bins')

# ── Energy helper (pure-NumPy fallback, works without numba) ──────────────────
def _calc_e_with_terms_np(s, params):
    """
    H(σ) = -Σ_ij J_ij σ_i σ_j  -  Σ_i h_i σ_i
    Returns (energy, J_term, h_term) each of shape (T,).
    """
    N    = s.shape[1]
    h_p  = params[:N]
    J_p  = params[N:]
    h_t  = s @ h_p                  # (T,)
    J_t  = np.zeros(s.shape[0])
    k = 0
    for ii in range(N - 1):
        for jj in range(ii + 1, N):
            J_t += J_p[k] * s[:, ii] * s[:, jj]
            k   += 1
    return -J_t - h_t, J_t, h_t

# Prefer the numba-compiled version from model.py if available
try:
    from model import calc_e, calc_e_with_terms as _calc_e_with_terms_nb
    def calc_e_with_terms(s, params):
        return _calc_e_with_terms_nb(s, params)
    print('Using numba calc_e_with_terms from model.py')
except Exception:
    calc_e_with_terms = _calc_e_with_terms_np
    print('numba not available — using pure-NumPy energy calculation')

# ── MCH fit ───────────────────────────────────────────────────────────────────
multipliers = None
model_sample = None
ising_ok = False

try:
    from coniii import MCH
    from coniii.samplers import Metropolis

    print(f'\nFitting inverse Ising model on {N_ISING} neurons, '
          f'{Nt} time bins, {args.n_cpus} CPUs …')
    t0_ising = tm()

    solver = MCH(
        spikes_pm,
        sample_size = args.sample_size,
        rng         = np.random.RandomState(args.seed),
        n_cpus      = args.n_cpus,
        sampler_kw  = {'boost': True},
    )

    def _learn(i):
        return {'maxdlamda': 1, 'eta': 1 / (i + 1)}

    multipliers = solver.solve(
        maxiter  = args.max_iter,
        n_iters  = max(N_ISING * 100, 2000),
        burn_in  = max(N_ISING * 50,  1000),
        iprint   = 'detailed',
        custom_convergence_f = _learn,
    )
    print(f'Ising fit done in {tm()-t0_ising:.1f} s')

    # Sample from the fitted model for quality-of-fit
    print('Sampling from fitted model …')
    t0_samp = tm()
    # calc_e must accept int64 array; use the numba version if available
    try:
        from model import calc_e as _calc_e_nb
        sampler = Metropolis(N_ISING, multipliers, _calc_e_nb)
    except Exception:
        # fallback: inline calc_e for Metropolis
        def _calc_e_fallback(s, p):
            N = s.shape[1]
            h = np.sum(s * p[:N], axis=1)
            J_sum = np.zeros(s.shape[0])
            k = 0
            for ii in range(N-1):
                for jj in range(ii+1, N):
                    J_sum += p[N+k] * s[:, ii] * s[:, jj]
                    k += 1
            return -J_sum - h
        sampler = Metropolis(N_ISING, multipliers, _calc_e_fallback)

    sampler.generate_sample_parallel_py(args.sample_size)
    model_sample = sampler.sample
    print(f'Sampling done in {tm()-t0_samp:.1f} s')
    ising_ok = True

except ImportError:
    print('\nWARNING: coniii not installed — Ising fit skipped.')
    print('  Run inside the singularity container to enable it.')
except Exception as exc:
    print(f'\nERROR during Ising fit: {exc}')
    import traceback; traceback.print_exc()

# ── Energy / term time series ─────────────────────────────────────────────────
if ising_ok and multipliers is not None:
    energies_t, J_terms_t, h_terms_t = calc_e_with_terms(
        spikes_pm, np.asarray(multipliers, dtype=np.float64))

    # Smooth with a causal Gaussian kernel scaled to the binned time axis
    sigma_b   = max(1, round(6 / args.bin_ms))        # ~6 ms in bin units
    s_b       = np.arange(-3*sigma_b, 3*sigma_b + 1)
    kern_b    = np.exp(-(s_b**2) / (2 * sigma_b**2))
    kern_b[s_b < 0] = 0
    kern_b   /= kern_b.sum()
    energy_s  = np.convolve(kern_b, energies_t, 'same')
    J_s       = np.convolve(kern_b, J_terms_t,  'same')
    h_s       = np.convolve(kern_b, h_terms_t,  'same')

    # Recurrent spans on the BINNED time axis (for energy/terms panels)
    rec_spans_binned, in_b = [], False
    for i in range(n_bins):
        if mode_binned[i] == 1 and not in_b:
            bs, in_b = time_binned[i], True
        elif mode_binned[i] == 0 and in_b:
            rec_spans_binned.append((bs, time_binned[i]));  in_b = False
    if in_b:
        rec_spans_binned.append((bs, time_binned[-1]))

    def shade_binned(ax):
        for (ts, te) in rec_spans_binned:
            ax.axvspan(ts, te, alpha=REC_ALPHA, color=REC_COLOR, zorder=0)
else:
    energy_s = J_s = h_s = None
    rec_spans_binned = []
    def shade_binned(ax): pass


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Figures
# ══════════════════════════════════════════════════════════════════════════════

legend_regime = [
    mpatches.Patch(color=REC_COLOR,   alpha=0.45, label='Collective regime'),
    mpatches.Patch(color='#CCCCCC',               label='Random (independent) regime'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Overview: raster / firing rate / energy / J-h terms
# ─────────────────────────────────────────────────────────────────────────────
n_panels      = 4 if ising_ok else 2
height_ratios = [3, 1.2, 1.2, 1.2][:n_panels]
fig_h         = 8 + 3 * (n_panels - 2)

fig1, axes1 = plt.subplots(
    n_panels, 1, figsize=(16, fig_h), sharex=True,
    gridspec_kw={'height_ratios': height_ratios},
)
ax_rast, ax_rate = axes1[0], axes1[1]
if ising_ok:
    ax_enrg, ax_term = axes1[2], axes1[3]

# — Raster —
shade(ax_rast)
shade_bursts(ax_rast)
ax_rast.plot(SeTimes, SeIdx,       'r.', ms=0.7, alpha=0.35, rasterized=True)
ax_rast.plot(SiTimes, SiIdx + Ne,  'b.', ms=0.7, alpha=0.35, rasterized=True)
ax_rast.axhline(Ne, color='k', lw=0.6, ls='--', alpha=0.4)
ax_rast.set_ylabel('Neuron index', fontsize=12)
ax_rast.set_ylim([0, Ne + Ni])
ax_rast.set_yticks([Ne // 2, Ne + Ni // 2])
ax_rast.set_yticklabels([f'{LBL_E}\n(0–{Ne})', f'{LBL_I}\n({Ne}–{Ne+Ni})'], fontsize=10)
ax_rast.set_title(
    f'Neural Activity — Random (independent) vs Collective (synchronized) regimes  '
    f'({SEG*dt:.0f} ms blocks,  {Nt} total steps = {T:.0f} ms)',
    fontsize=12, pad=8)
ax_rast.legend(handles=[
    mpatches.Patch(color=REC_COLOR,   alpha=0.45, label='Collective block'),
    mpatches.Patch(color=BURST_COLOR, alpha=0.50, label='Synchronized burst event'),
    mpatches.Patch(color='#CCCCCC',               label='Random (independent)'),
    mpatches.Patch(color='red',       alpha=0.6,  label=LBL_E),
    mpatches.Patch(color='blue',      alpha=0.6,  label=LBL_I),
], loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

# — Firing rate —
shade(ax_rate)
ax_rate.plot(time_ax, 1000*re, color='red',  lw=1.6, label=LBL_E)
ax_rate.plot(time_ax, 1000*ri, color='blue', lw=1.6, label=LBL_I)
ax_rate.set_ylabel('Firing rate (Hz)', fontsize=12)
ax_rate.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax_rate.set_xlim([0, T])

if ising_ok:
    # — Energy H(t) = -J_term - h_term —
    shade_binned(ax_enrg)
    ax_enrg.plot(time_binned, energy_s, color='#1a1a2e', lw=1.6)
    ax_enrg.set_ylabel('Energy  H(t)', fontsize=12)
    ax_enrg.set_xlim([0, T])
    ax_enrg.set_title(
        f'Ising Hamiltonian  H(t)  —  N={N_ISING} {LBL_E},  '
        f'{args.bin_ms:.1f} ms bins,  {n_bins} samples',
        fontsize=11, pad=4)
    ax_enrg.legend(handles=legend_regime, loc='upper right',
                   fontsize=9, framealpha=0.9)

    # — J_term and h_term —
    shade_binned(ax_term)
    ax_term.plot(time_binned, J_s, color='#e63946', lw=1.6,
                 label=r'$J$-term  (pairwise correlations — rises in collective/reach)')
    ax_term.plot(time_binned, h_s, color='#457b9d', lw=1.6,
                 label=r'$h$-term  (local fields — dominates in random/pre-reach)')
    ax_term.set_xlabel('Time (ms)', fontsize=12)
    ax_term.set_ylabel('Term contribution', fontsize=12)
    ax_term.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_term.set_xlim([0, T])
    ax_term.set_title(
        r'Hamiltonian decomposition  $H = -J\mathrm{-term} - h\mathrm{-term}$  '
        r'[J jumps when neurons synchronize during reach]',
        fontsize=11, pad=4)

    ax_rate.set_xlabel('')
else:
    ax_rate.set_xlabel('Time (ms)', fontsize=12)

plt.tight_layout(h_pad=0.4)
p1 = os.path.join(OUT_DIR, 'spike_activity_overview.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f'Saved → {p1}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Detailed: separate E/I rasters + firing rate
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 11), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1.2]})

shade(axes2[0])
axes2[0].plot(SeTimes, SeIdx, 'r.', ms=0.7, alpha=0.35, rasterized=True)
axes2[0].set_ylabel(f'{LBL_E} index', fontsize=11)
axes2[0].set_ylim([0, Ne])
axes2[0].set_title(f'{LBL_E} Spike Raster', fontsize=12)
axes2[0].legend(handles=legend_regime, loc='upper right', fontsize=9)

shade(axes2[1])
axes2[1].plot(SiTimes, SiIdx, 'b.', ms=1.0, alpha=0.45, rasterized=True)
axes2[1].set_ylabel(f'{LBL_I} index', fontsize=11)
axes2[1].set_ylim([0, Ni])
axes2[1].set_title(f'{LBL_I} Spike Raster', fontsize=12)

shade(axes2[2])
axes2[2].plot(time_ax, 1000*re, color='red',  lw=1.6, label=LBL_E)
axes2[2].plot(time_ax, 1000*ri, color='blue', lw=1.6, label=LBL_I)
axes2[2].set_xlabel('Time (ms)', fontsize=12)
axes2[2].set_ylabel('Firing rate (Hz)', fontsize=12)
axes2[2].legend(loc='upper right', fontsize=9)
axes2[2].set_title('Population Firing Rate', fontsize=12)
axes2[2].set_xlim([0, T])

plt.tight_layout(h_pad=0.5)
p2 = os.path.join(OUT_DIR, 'spike_raster_detailed.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved → {p2}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Per-block mean firing rates
# ─────────────────────────────────────────────────────────────────────────────
n_blocks     = TOTAL_STEPS // SEG
re_blk       = [1000 * np.mean(re_raw[b*SEG:(b+1)*SEG]) for b in range(n_blocks)]
ri_blk       = [1000 * np.mean(ri_raw[b*SEG:(b+1)*SEG]) for b in range(n_blocks)]
blk_regime   = [mode[b*SEG] for b in range(n_blocks)]
bar_colors   = [REC_COLOR if r == 1 else '#888888' for r in blk_regime]
xtlabels     = [f'B{b+1}\n{"Col" if blk_regime[b] else "Rnd"}\n'
                f'{b*SEG*dt:.0f}–{(b+1)*SEG*dt:.0f}ms'
                for b in range(n_blocks)]
x = np.arange(n_blocks)

fig3, (ax3e, ax3i) = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, title in zip((ax3e, ax3i), (re_blk, ri_blk),
                           (LBL_E, LBL_I)):
    ax.bar(x, data, color=bar_colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(x);  ax.set_xticklabels(xtlabels, fontsize=6.5)
    ax.set_ylabel('Mean firing rate (Hz)');  ax.set_xlabel('Block')
    ax.set_title(f'{title} neurons — mean rate per block', fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color=REC_COLOR, label='Collective'),
        mpatches.Patch(color='#888888', label='Random'),
    ], fontsize=9)

plt.tight_layout()
p3 = os.path.join(OUT_DIR, 'block_firing_rates.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f'Saved → {p3}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Quality of fit  (requires coniii)
# ─────────────────────────────────────────────────────────────────────────────
p4 = None
if ising_ok and model_sample is not None:
    try:
        from visualization import plot_model_quality, plot_model_quality_summary

        print('\nGenerating quality-of-fit figures …')
        # model_quality_summary.png — pairwise/triplet correlations, P(K), J-dist
        plot_model_quality_summary(
            spikes_pm, model_sample, multipliers, N_ISING, OUT_DIR)
        p4 = os.path.join(OUT_DIR, 'model_quality_summary.png')
        print(f'Saved → {p4}')

        # correlation_order_*.png — k-order scatter plots
        plot_model_quality(spikes_pm, model_sample, OUT_DIR)
        print(f'Saved correlation_order plots → {OUT_DIR}/')

    except Exception as exc:
        print(f'WARNING: quality-of-fit plot failed: {exc}')
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Text summary
# ══════════════════════════════════════════════════════════════════════════════
rand_m = (mode == 0);  rec_m = (mode == 1)
re_rand = 1000 * np.mean(re_raw[rand_m]);  re_rec = 1000 * np.mean(re_raw[rec_m])
ri_rand = 1000 * np.mean(ri_raw[rand_m]);  ri_rec = 1000 * np.mean(ri_raw[rec_m])
n_rand_blk = sum(1 for b in range(n_blocks) if blk_regime[b] == 0)
n_col_blk  = sum(1 for b in range(n_blocks) if blk_regime[b] == 1)

ising_lines = []
if ising_ok and multipliers is not None:
    h_params    = np.asarray(multipliers[:N_ISING])
    J_params    = np.asarray(multipliers[N_ISING:])
    rand_b_mask = (mode_binned == 0)
    rec_b_mask  = (mode_binned == 1)
    ising_lines = [
        '',
        f'Ising model  (N={N_ISING} neurons):',
        f'  Spike bins  : {n_bins}  ({args.bin_ms:.1f} ms each, {BIN_STEPS} raw steps)',
        f'  ⟨σ⟩ range   : [{mean_spins.min():.3f}, {mean_spins.max():.3f}]',
        f'  h params    : mean={h_params.mean():.4f}  std={h_params.std():.4f}',
        f'  J params    : mean={J_params.mean():.4f}  std={J_params.std():.4f}',
        f'  Energy      : random {np.mean(energies_t[rand_b_mask]):.3f}'
                        f'  collective {np.mean(energies_t[rec_b_mask]):.3f}',
        f'  J-term      : random {np.mean(J_terms_t[rand_b_mask]):.3f}'
                        f'  collective {np.mean(J_terms_t[rec_b_mask]):.3f}',
        f'  h-term      : random {np.mean(h_terms_t[rand_b_mask]):.3f}'
                        f'  collective {np.mean(h_terms_t[rec_b_mask]):.3f}',
    ]
else:
    ising_lines = ['', 'Ising model: not run (coniii unavailable)']

outputs = [p1, p2, p3] + ([p4] if p4 else [])
summary_lines = [
    'Neural Simulation + Ising Analysis Summary',
    '=' * 50,
    f'Total time          : {T:.0f} ms  ({Nt} steps, dt={dt} ms)',
    f'Network             : {Ne} principal ({LBL_E}) + {Ni} interneurons ({LBL_I})',
    f'Regime blocks       : {n_rand_blk} random | {n_col_blk} collective  '
    f'({SEG} steps = {SEG*dt:.0f} ms each)',
    f'Collective regime   : {args.burst_rate:.0f} Hz burst / {args.bg_rate:.0f} Hz quiet  '
    f'(burst frac ≈ {burst_frac:.2f},  mean ≈ {mean_col_hz:.0f} Hz)',
    f'Random regime       : {args.rand_rate:.0f} Hz independent Poisson',
    '',
    'Mean firing rates:',
    f'  {LBL_E}  Random={re_rand:.2f} Hz   Collective={re_rec:.2f} Hz',
    f'  {LBL_I}  Random={ri_rand:.2f} Hz   Collective={ri_rec:.2f} Hz',
] + ising_lines + [
    '',
    'Outputs:',
] + [f'  {os.path.basename(p)}' for p in outputs if p]

summary = '\n'.join(summary_lines)
sp = os.path.join(OUT_DIR, 'summary.txt')
with open(sp, 'w') as f:
    f.write(summary)

print('\n' + summary)
print(f'\nSaved → {sp}')
print('\nDone.')
