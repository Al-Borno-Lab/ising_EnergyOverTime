#!/bin/bash
# E10 step 2 CONTINGENCY worker (split N=19 conditions). Usage: sbatch [opts] step2_split_job.sbatch <npz> <outdir> <keys>
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
set -euo pipefail
NPZ=$1; OUT=$2; KEYS=$3; shift 3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
cd "${E10_CODE:-$HOME/ising_wt/e10/10_operating_point_h2_h4/code}"
mkdir -p "$OUT"
singularity exec "$SIF" /entrypoint.sh python "$PWD/step2_split.py" --npz "$NPZ" --out "$OUT" --ncpu "${SLURM_CPUS_PER_TASK:-1}" --keys "$KEYS" "$@"
