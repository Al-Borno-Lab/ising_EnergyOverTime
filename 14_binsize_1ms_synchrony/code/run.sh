#!/usr/bin/env bash
# Entry point for E14 (bin-size sweep and 1-ms synchrony; Nashef reconciliation). Run from the repo root on the cluster.
# Order (PROTOCOL.md): extraction + gates -> synthetic validation must pass and stats_e14.py be reviewed -> read real results.
set -euo pipefail
EXP="14_binsize_1ms_synchrony"
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
mkdir -p "$EXP/results" "$EXP/logs"
case "${1:-submit}" in
  extract) sbatch $EXP/code/sbatch_extract.sh ;;     # step 1 (done: job 4363528, gates G1-G3 pass 21/21)
  submit)
    s=$(sbatch --parsable -J e14syn --array=0-3 -o $EXP/logs/syn_%A_%a.out $EXP/code/e14.sbatch synth)
    r=$(sbatch --parsable -J e14real --array=0-20 -o $EXP/logs/real_%A_%a.out $EXP/code/e14.sbatch real)
    echo "synth $s real $r" ;;
  stats)
    $PY $EXP/code/stats_e14.py synth $EXP/results/synth $EXP/assets
    $PY $EXP/code/stats_e14.py real  $EXP/results/real  $EXP/assets ;;
esac
