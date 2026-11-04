#!/bin/bash
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
# From the repo root:  sbatch --array=0-20 e16.sbatch real   |   sbatch --array=0-5 e16.sbatch synth
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EXP=16_onems_timing; SPK=${E16_SPK:-$HOME/ising_wt/e14/14_binsize_1ms_synchrony/results/spk}
LMK=$PWD/09_claude_science_handoff/assets/tables/phase_landmarks.csv
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
SESS=(210421 210422 210423 210425 210511 210512 210514 210515 210606 210608 210614 210619 210620 210622 210623 220515 220516 220517 220518 220519 220520)
cd $EXP/code
case "$1" in
  real)  $PY run_session.py $SPK $LMK ../results/real ${SESS[$SLURM_ARRAY_TASK_ID]} $SLURM_CPUS_PER_TASK ;;
  synth) $PY synth_e16.py   $SPK $LMK ../results/synth $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK ;;
esac
