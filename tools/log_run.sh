#!/usr/bin/env bash
# Append a run record to <experiment>/runs/RUN_LOG.md. Call it for every submission
# (sbatch or local) BEFORE results are looked at, and again with the outcome.
#
#   bash tools/log_run.sh <experiment_dir> submitted "<command>" [slurm_job_id] [output_dir]
#   bash tools/log_run.sh <experiment_dir> finished  "<one-line outcome>" [slurm_job_id]
#   bash tools/log_run.sh <experiment_dir> failed    "<what broke>"       [slurm_job_id]
set -euo pipefail

exp="${1:?usage: log_run.sh <experiment_dir> <submitted|finished|failed> \"<text>\" [job_id] [output_dir]}"
event="${2:?missing event}"
text="${3:?missing text}"
job="${4:--}"
out="${5:--}"
case "$event" in submitted|finished|failed) ;; *) echo "event must be submitted|finished|failed" >&2; exit 1;; esac

log="$exp/runs/RUN_LOG.md"
[ -f "$log" ] || { echo "no $log (is $exp an experiment folder?)" >&2; exit 1; }

commit=$(git -C "$exp" rev-parse --short HEAD)
dirty=""
git -C "$exp" diff --quiet HEAD -- . || dirty=" (+uncommitted changes in experiment)"
printf '| %s | %s | %s | %s%s | %s | `%s` | %s |\n' \
  "$(date '+%Y-%m-%d %H:%M')" "$event" "$(hostname -s)" "$commit" "$dirty" "$job" "$out" "${text//|/\\|}" >> "$log"
echo "logged $event in $log"
