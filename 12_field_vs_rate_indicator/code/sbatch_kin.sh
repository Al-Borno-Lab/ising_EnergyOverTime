#!/usr/bin/env bash
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J e12_kin
# usage: sbatch sbatch_kin.sh <dir containing raw/ and code/>   (raw files copied under $HOME beforehand)
D="$1"
singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python "$D/extract_kinematics.py" "$D/raw" "$D/kinematics.npz"
