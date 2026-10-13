#!/bin/bash
#SBATCH --account=ceas-alborno
#SBATCH --partition=math-alderaan
#SBATCH --job-name=e11_laser
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# E11 step 1 (rerun at full precision, amendment A3): copy the 21 raw files, extract ReachS fields, merge with spikes_v2, delete the copies.
set -euo pipefail
cd "$HOME/ising_wt/e11_repo/11_stimulation_kappa"
SRC=/data001/projects/enserrog/dylan/AbigailUpdated; RAW=results/raw_tmp; SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
SPK="$HOME/ising_EnergyOverTime/energy_over_time_organized/09_claude_science_handoff/data/spikes_v2.npz"
mkdir -p $RAW
for f in $(ls $SRC | grep -v 'WT2\|220519_DJC000\|220520_DJC002'); do cp -n $SRC/$f $RAW/; done
echo "code commit $(git rev-parse --short HEAD)"
singularity exec $SIF /entrypoint.sh python code/extract_laser_onsets.py $RAW results/reachs_raw_v2.csv
singularity exec $SIF /entrypoint.sh python code/merge_laser_onsets.py results/reachs_raw_v2.csv "$SPK" results/laser_onsets_v2.csv
rm -rf $RAW; echo "raw copies deleted"
