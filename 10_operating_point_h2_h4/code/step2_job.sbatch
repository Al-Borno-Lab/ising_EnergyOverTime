#!/bin/bash
# E10 step 2 worker. Usage: sbatch [opts] step2_job.sbatch <npz> <outdir> <keys|all> <nshuf> [extra args]
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
set -euo pipefail
NPZ=$1; OUT=$2; KEYS=$3; NSHUF=$4; shift 4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
# sbatch runs a spooled copy of this script, so locate the code via the submit directory (submit from the repo root)
cd "$SLURM_SUBMIT_DIR/10_operating_point_h2_h4/code"
singularity exec "$SIF" /entrypoint.sh python "$PWD/step2_real_h2_h4.py" --npz "$NPZ" --out "$OUT" --ncpu "${SLURM_CPUS_PER_TASK:-1}" --keys "$KEYS" --nshuf "$NSHUF" "$@"
