#!/usr/bin/env bash
# Entry point for E12 (Field-term dip vs firing rate as a marker of peak acceleration). Reproduces every output of this experiment.
# Run from the repo root on the cluster. Heavy steps must go through sbatch.
set -euo pipefail
EXP="12_field_vs_rate_indicator"
SIF="$HOME/projectDir/singularity-env/inverse-ising-arm-2.sif"
PY="singularity exec $SIF /entrypoint.sh python"

mkdir -p "$EXP/results"
# Example step (replace):
# job=$(sbatch --parsable -A ceas-alborno -p math-alderaan --cpus-per-task=32 --mem=128G -t 12:00:00 \
#        --wrap "$PY $EXP/code/step1.py --out $EXP/results/step1")
# bash tools/log_run.sh "$EXP" submitted "step1" "$job" "$EXP/results/step1"
echo "Edit $EXP/code/run.sh to define the steps." >&2
exit 1
