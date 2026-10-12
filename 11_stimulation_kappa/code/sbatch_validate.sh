#!/bin/bash
#SBATCH --account=ceas-alborno
#SBATCH --partition=math-alderaan
#SBATCH --job-name=e11_validate
#SBATCH --cpus-per-task=60
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# Synthetic validation of stimfit_model (criteria: VALIDATION_CRITERIA_2026-10-12.md). Run from ~/ising_wt/e11.
cd "$HOME/ising_wt/e11"
singularity exec "$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif" /entrypoint.sh \
  python code/validate_stimfit.py 60 "${1:-results/validation}" 12 60:8,18:16
