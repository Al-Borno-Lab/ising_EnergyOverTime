#!/bin/bash
# E10 step 2: one task per array element, parallel inside the task. Submit with the task list prepared first:
#   python step2_task.py --npz $NPZ --out $OUT --skip-dirs <old step2 dir> --exclude-keys <N=19 keys> --list-tasks --tasklist LIST
#   sbatch --array=0-(n-1)%20 --cpus-per-task=16 --mem=40G -t 03:00:00 step2_task_array.sbatch NPZ OUT LIST
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
set -euo pipefail
NPZ=$1; OUT=$2; LIST=$3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
cd "${E10_CODE:-$HOME/ising_wt/e10/10_operating_point_h2_h4/code}"
singularity exec "$SIF" /entrypoint.sh python "$PWD/step2_task.py" --npz "$NPZ" --out "$OUT" --ncpu "${SLURM_CPUS_PER_TASK:-16}" --index "$SLURM_ARRAY_TASK_ID" --tasklist "$LIST" --skip-dirs "${E10_SKIP:-}"
