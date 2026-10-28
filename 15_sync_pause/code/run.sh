#!/usr/bin/env bash
# Entry point for E15 (synchronous pauses: shared drive vs fixed coupling). Run from the repo root on the cluster.
# Order (PROTOCOL.md): library check + synthetic validation must pass, and stats_e15.py must be reviewed,
# before the real-data results are read.
set -euo pipefail
EXP="15_sync_pause"
PY="singularity exec $HOME/projectDir/singularity-env/inverse-ising-arm-2.sif /entrypoint.sh python"
mkdir -p "$EXP/results" "$EXP/logs"
case "${1:-submit}" in
  submit)
    v=$(sbatch --parsable -J e15val -o $EXP/logs/val_%j.out $EXP/code/e15.sbatch validate)
    s=$(sbatch --parsable -J e15syn --array=0-5 -o $EXP/logs/syn_%A_%a.out $EXP/code/e15.sbatch synth)
    r=$(sbatch --parsable -J e15real --array=0-20 -o $EXP/logs/real_%A_%a.out $EXP/code/e15.sbatch real)
    for j in "validate:$v" "synth:$s" "real:$r"; do bash tools/log_run.sh "$EXP" submitted "${j%%:*}" "${j##*:}" "$EXP/results"; done ;;
  stats)   # light; login node is fine
    $PY $EXP/code/stats_e15.py synth $EXP/results/synth $EXP/assets
    $PY $EXP/code/stats_e15.py real  $EXP/results/real  $EXP/assets ;;
esac
