#!/bin/bash
#SBATCH --job-name=neural_sim
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=./logs/neural_sim/neural_sim_%j.log

# ── Quick-start examples ──────────────────────────────────────────────────────
#
# Default Purkinje cell simulation (5 000 steps, 200 ms blocks):
#   sbatch run_neural_simulation_sbatch.sh \
#       --label_e "Purkinje cells" --label_i "Basket/Stellate" \
#       --ne 100 --ni 25
#
# Better Ising fit (coarser 5 ms bins, 50 000 steps → 10 000 Ising samples):
#   sbatch run_neural_simulation_sbatch.sh \
#       --total_steps 50000 --bin_ms 5 \
#       --n_ising 15 --sample_size 100000 --max_iter 200 \
#       --label_e "Purkinje cells" --label_i "Basket/Stellate" --ne 100 --ni 25
#
# Adjust synchrony pattern (longer bursts, higher burst rate):
#   sbatch run_neural_simulation_sbatch.sh \
#       --burst_rate 250 --bg_rate 10 --p_burst_on 0.003 --p_burst_off 0.01


set -euo pipefail

# ── Container ─────────────────────────────────────────────────────────────────
CONTAINER="~/projectDir/singularity-env/inverse-ising-arm-2.sif"
CONTAINER_EXPANDED="${CONTAINER/#\~/${HOME}}"

# ── Resolve the directory that contains simulate_neural_regimes.py ────────────
_resolve_script_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && \
       [[ -f "${SLURM_SUBMIT_DIR}/simulate_neural_regimes.py" ]]; then
        (cd "${SLURM_SUBMIT_DIR}" && pwd)
        return
    fi
    local _s="${BASH_SOURCE[0]}"
    [[ "${_s}" != /* ]] && _s="${PWD}/${_s}"
    command -v readlink >/dev/null 2>&1 && \
        readlink -f / >/dev/null 2>&1 && \
        _s="$(readlink -f "${_s}")"
    (cd "$(dirname "${_s}")" && pwd)
}

SCRIPT_DIR="$(_resolve_script_dir)"

# ── Parse optional arguments ──────────────────────────────────────────────────
# Usage (run from energy_over_time_script/):
#   sbatch run_neural_simulation_sbatch.sh [OPTIONS]
#
# Simulation model:
#   RANDOM  blocks  — each neuron fires independently at --rand_rate Hz (default 60 Hz)
#                     No pairwise correlations  →  h-term dominates Ising Hamiltonian.
#   COLLECTIVE blocks — shared burst/quiet state (Markov chain) drives co-firing.
#                     Bursts at --burst_rate Hz (200 Hz default), quiet at --bg_rate Hz.
#                     Strong pairwise correlations  →  J-term jumps (mimics PC reach).
#
# Key parameters:
#   --rand_rate F      : pre-reach Poisson rate Hz (default 60)
#   --burst_rate F     : burst firing rate Hz in collective mode (default 200)
#   --bg_rate F        : quiet background rate Hz in collective mode (default 20)
#   --p_burst_on F     : quiet→burst transition prob/step (default 0.005 → ~20 ms quiet)
#   --p_burst_off F    : burst→quiet transition prob/step (default 0.02  → ~5 ms bursts)
#   --ne N / --ni N    : population sizes (default 800/200; use 100/25 for Purkinje)
#   --bin_ms F         : spike bin width for Ising (default 0.1; use 5.0 for better fit)
#   --n_ising N        : neurons to fit Ising on (default 20; fewer = faster/better)

EXTRA_ARGS=("$@")

# ── Log directory ─────────────────────────────────────────────────────────────
mkdir -p ./logs/neural_sim

# ── Announce ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  neural_sim job  ${SLURM_JOB_ID:-local}"
echo "  script dir  : ${SCRIPT_DIR}"
echo "  container   : ${CONTAINER_EXPANDED}"
echo "  extra args  : ${EXTRA_ARGS[*]:-<none>}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK:-16}"
echo "============================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

singularity exec "${CONTAINER_EXPANDED}" /entrypoint.sh \
    python simulate_neural_regimes.py "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "  neural_sim job ${SLURM_JOB_ID:-local} finished."
echo "  Outputs → ${SCRIPT_DIR}/notes/neural_simulation/"
echo "============================================================"
