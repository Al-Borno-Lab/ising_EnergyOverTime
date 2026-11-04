#!/usr/bin/env bash
# Entry point for E16. Run from the repo root on the cluster. Input: E14's 1-ms extraction (gates G1-G3 pass 21/21), path in E16_SPK.
set -euo pipefail
EXP="16_onems_timing"; mkdir -p $EXP/results $EXP/logs
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
case "${1:-submit}" in
  submit)
    s=$(sbatch --parsable -J e16syn --array=0-5 -o $EXP/logs/syn_%A_%a.out $EXP/code/e16.sbatch synth)
    r=$(sbatch --parsable -J e16real --array=0-20 -o $EXP/logs/real_%A_%a.out $EXP/code/e16.sbatch real)
    echo "synth $s real $r" ;;
  stats)
    $PY $EXP/code/stats_e16.py synth $EXP/results/synth $EXP/assets
    $PY $EXP/code/stats_e16.py real  $EXP/results/real  $EXP/assets ;;
esac
