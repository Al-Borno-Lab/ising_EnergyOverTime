#!/usr/bin/env bash
# Create a new experiment folder from _template/.
#   bash tools/new_experiment.sh <short_name> "<Title>" <owner>
# Example:
#   bash tools/new_experiment.sh operating_point_h2_h4 "Operating point H2-H4 on real data" grad-researcher
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
name="${1:?usage: new_experiment.sh <short_name> \"<Title>\" <owner>}"
title="${2:?missing title}"
owner="${3:?missing owner}"

last=$(ls -d "$ROOT"/[0-9][0-9]_* 2>/dev/null | xargs -n1 basename | cut -c1-2 | sort -n | tail -1)
next=$(printf "%02d" $((10#${last:-0} + 1)))
dir="$ROOT/${next}_${name}"
[ -e "$dir" ] && { echo "exists: $dir" >&2; exit 1; }

cp -R "$ROOT/_template" "$dir"
today=$(date +%Y-%m-%d)
for f in "$dir"/README.md "$dir"/RESULTS.md "$dir"/experiment.json "$dir"/runs/RUN_LOG.md "$dir"/code/run.sh; do
  sed -i.bak -e "s/{{ID}}/E${next}/g" -e "s/{{TITLE}}/${title//\//\\/}/g" \
             -e "s/{{OWNER}}/${owner}/g" -e "s/{{DATE}}/${today}/g" -e "s/{{DIR}}/${next}_${name}/g" "$f"
  rm -f "$f.bak"
done
chmod +x "$dir/code/run.sh"
python3 "$ROOT/tools/update_status.py" || true
echo "Created $dir"
echo "Next: fill in README.md (design) and experiment.json, then commit before running anything."
