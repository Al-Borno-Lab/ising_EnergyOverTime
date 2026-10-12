#!/usr/bin/env bash
# generate_paper_figures.sh
# --------------------------------------------------------------------------
# Rapidly (re)generate every session-comparison figure needed for the paper,
# in one command. Each figure is written as PDF (drop straight into the
# paper) + PNG (quick preview), with panels already labeled A1-D1/A2-D2 (or
# A1-E1/A2-E2 for pk_analysis) -- no Illustrator relabeling step needed. Just
# rerun this after regenerating any upstream CSVs and every comparison below
# gets rebuilt in place with the same filenames.
#
# Edit the COMPARISONS array below to match what the paper actually needs:
# one line per comparison, columns are
#   session1  stim1  session2  stim2  [label1]  [label2]
# label1/label2 are optional (default "Session <id>, stim <n>" if omitted or
# set to "-"); use them for things like "Baseline"/"Perturbation" instead of
# raw session IDs. Add/remove/edit rows freely.
# --------------------------------------------------------------------------
set -uo pipefail

DATA_FOLDER="outputs/energy_decomp_Jul_10_svg"
OUTPUT_DIR="$DATA_FOLDER/comparisons"

# session1  stim1  session2  stim2  label1        label2
COMPARISONS=(
  "210421    0     210422    0     -             -"
  "210421    0     210421    1     -             -"
  "210515    1     220515    1     -             -"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

n_total=0
n_failed=0
failed_pairs=()

for row in "${COMPARISONS[@]}"; do
    read -r session1 stim1 session2 stim2 label1 label2 <<< "$row"
    n_total=$((n_total + 1))

    extra_args=()
    [[ -n "${label1:-}" && "$label1" != "-" ]] && extra_args+=(--label1 "$label1")
    [[ -n "${label2:-}" && "$label2" != "-" ]] && extra_args+=(--label2 "$label2")

    echo "=== [$n_total] $session1 (stim $stim1)  vs  $session2 (stim $stim2) ==="
    python compare_sessions.py \
        --data_folder "$DATA_FOLDER" \
        --session1 "$session1" --stim1 "$stim1" \
        --session2 "$session2" --stim2 "$stim2" \
        --output_dir "$OUTPUT_DIR" \
        "${extra_args[@]}"

    if [[ $? -ne 0 ]]; then
        n_failed=$((n_failed + 1))
        failed_pairs+=("$session1/stim$stim1 vs $session2/stim$stim2")
    fi
    echo
done

echo "=================================================================="
echo "Done: $((n_total - n_failed))/$n_total comparisons succeeded."
if [[ $n_failed -gt 0 ]]; then
    echo "Failed:"
    for p in "${failed_pairs[@]}"; do
        echo "  - $p"
    done
    exit 1
fi
