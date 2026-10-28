#!/bin/bash
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
# From the repo root:  sbatch --array=0-20 -J e14real 14_binsize_1ms_synchrony/code/e14.sbatch real
#                      sbatch --array=0-3  -J e14syn  14_binsize_1ms_synchrony/code/e14.sbatch synth
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
EXP=14_binsize_1ms_synchrony; C=$EXP/code; SPK=$EXP/results/spk
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
SESS=(210421 210422 210423 210425 210511 210512 210514 210515 210606 210608 210614 210619 210620 210622 210623 220515 220516 220517 220518 220519 220520)
cd $C
case "$1" in
  real)  $PY run_session.py ../results/spk ../results/real ${SESS[$SLURM_ARRAY_TASK_ID]} $SLURM_CPUS_PER_TASK ;;
  synth) $PY synth_e14.py   ../results/spk ../results/synth $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK ;;
esac
