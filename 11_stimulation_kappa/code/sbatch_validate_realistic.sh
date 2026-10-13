#!/bin/bash
#SBATCH --account=ceas-alborno
#SBATCH --partition=math-alderaan
#SBATCH --job-name=e11_valreal
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
# usage: sbatch code/sbatch_validate_realistic.sh <out_dir> ; run from the clone ~/ising_wt/e11_repo/11_stimulation_kappa
cd "$HOME/ising_wt/e11_repo/11_stimulation_kappa"
echo "code commit $(git rev-parse --short HEAD)"
singularity exec "$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif" /entrypoint.sh python code/validate_realistic.py \
  "$HOME/ising_EnergyOverTime/energy_over_time_organized/09_claude_science_handoff/data/spikes_v2.npz" results/laser_onsets_v2.csv "${1:-results/validation_run6}" 60 8
