#!/bin/bash
#SBATCH --account=ceas-alborno
#SBATCH --partition=math-alderaan
#SBATCH --job-name=e11_run
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# usage: sbatch [--cpus-per-task=N] [--time=..] code/sbatch_run.sh <out_dir> <sessions|all> [extra run_stimfit.py flags...]
# Run from the repo clone on the cluster (~/ising_wt/e11_repo/11_stimulation_kappa). One BLAS thread per worker process.
cd "$HOME/ising_wt/e11_repo/11_stimulation_kappa"
OUT="$1"; SESS="$2"; shift 2
echo "code commit: $(git rev-parse --short HEAD)"
singularity exec "$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif" /entrypoint.sh \
  python code/run_stimfit.py "$HOME/ising_EnergyOverTime/energy_over_time_organized/09_claude_science_handoff/data/spikes_v2.npz" \
  assets/tables/laser_onsets.csv "$OUT" "${SLURM_CPUS_PER_TASK:-8}" "$SESS" "$@"
