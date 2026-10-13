#!/bin/bash
# E10 sensitivity worker (real fits only). Usage: sbatch [opts] step2_sens_job.sbatch <npz> <outdir>   (runs modes subset16, allreaches, late)
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
set -euo pipefail
NPZ=$1; OUT=$2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
cd "${E10_CODE:-$HOME/ising_wt/e10/10_operating_point_h2_h4/code}"
for MODE in subset16 allreaches late; do
  singularity exec "$SIF" /entrypoint.sh python "$PWD/step2_sens.py" --npz "$NPZ" --out "$OUT" --ncpu "${SLURM_CPUS_PER_TASK:-1}" --mode $MODE
done
