#!/bin/bash
#SBATCH -A ceas-alborno
#SBATCH -p math-alderaan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH -t 04:00:00
# Usage (from the repo root): sbatch --array=0-20 -J e15real 15_sync_pause/code/e15.sbatch real
#                             sbatch --array=0-5  -J e15syn  15_sync_pause/code/e15.sbatch synth
#                             sbatch              -J e15val  15_sync_pause/code/e15.sbatch validate
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EXP=15_sync_pause; C=$EXP/code; RES=$EXP/results
DATA=${E15_SPIKES:-$HOME/e10_data/spikes_v2.npz}            # container cannot see /data001: copy under $HOME
LMK=09_claude_science_handoff/assets/tables/phase_landmarks.csv
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
SESS=(210421 210422 210423 210425 210511 210512 210514 210515 210606 210608 210614 210619 210620 210622 210623 220515 220516 220517 220518 220519 220520)
case "$1" in
  real)     $PY $C/run_session.py $DATA $LMK $RES/real ${SESS[$SLURM_ARRAY_TASK_ID]} ;;
  synth)    $PY $C/synth_e15.py   $DATA $LMK $RES/synth $SLURM_ARRAY_TASK_ID ;;
  validate) mkdir -p $RES && $PY $C/validate_lib.py $DATA $LMK $RES/validate_lib.csv ;;
esac
