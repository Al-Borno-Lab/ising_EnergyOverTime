#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_decision_tree_pipeline.sh
#
# Builds the per-session/stim dataset and runs the decision tree in one go.
# All parameters are optional — defaults match the standard analysis setup.
#
# Usage:
#   bash run_decision_tree_pipeline.sh [options]
#
# Options:
#   --data_folder PATH      Root data directory          (default: energy_decomp_Apr_16)
#   --window LO HI          Search window in time bins   (default: 350 475)
#   --stim_min N            First stim index             (default: 0)
#   --stim_max N            Last stim index exclusive    (default: 3)
#   --peak_threshold F      J peak detection threshold   (default: 1.75)
#   --sessions ID ...       Restrict to specific sessions (default: all)
#   --run_name NAME         Label for the output folder  (default: auto-generated)
#   --max_depth N           Cap decision tree depth      (default: uncapped)
#   --no_cv                 Skip cross-validation        (default: run CV)
#   --exclude_cols COL ...  Feature columns to exclude   (default: none)
#   --help                  Show this message
#
# Examples:
#   # All sessions, default window
#   bash run_decision_tree_pipeline.sh
#
#   # Exclude circular J features, cap depth at 3
#   bash run_decision_tree_pipeline.sh \
#       --max_depth 3 --no_cv \
#       --exclude_cols std_j_in_window mean_j_in_window \
#       --run_name depth3_no_j_stats
#
#   # Selected sessions only
#   bash run_decision_tree_pipeline.sh \
#       --sessions 210425 210511 210515 220515 220516 220517 220518 220519 220520 \
#       --run_name manual_sessions
# ---------------------------------------------------------------------------

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_FOLDER="/data001/projects/enserrog/AbigailData/energy_over_time/energy_decomp_may12_stimDecon/"
WIN_LO=350
WIN_HI=475
STIM_MIN=0
STIM_MAX=3
PEAK_THRESHOLD=1.75
SESSIONS=""
RUN_NAME=""
MAX_DEPTH="3"
NO_CV=0
EXCLUDE_COLS="std_j_in_window mean_j_in_window"

DATASET_BASE="./notes/decision_dataset"

# ── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_folder)    DATA_FOLDER="$2";  shift 2 ;;
        --window)         WIN_LO="$2"; WIN_HI="$3"; shift 3 ;;
        --stim_min)       STIM_MIN="$2";     shift 2 ;;
        --stim_max)       STIM_MAX="$2";     shift 2 ;;
        --peak_threshold) PEAK_THRESHOLD="$2"; shift 2 ;;
        --run_name)       RUN_NAME="$2";     shift 2 ;;
        --max_depth)      MAX_DEPTH="$2";    shift 2 ;;
        --no_cv)          NO_CV=1;           shift   ;;
        --sessions)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SESSIONS="$SESSIONS $1"; shift
            done
            ;;
        --exclude_cols)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXCLUDE_COLS="$EXCLUDE_COLS $1"; shift
            done
            ;;
        --help)
            sed -n '2,30p' "$0"   # print the header comment
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Auto-generate run name if not supplied ───────────────────────────────────
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="w${WIN_LO}_${WIN_HI}_stim${STIM_MIN}-${STIM_MAX}_thr${PEAK_THRESHOLD}"
    [[ -n "$SESSIONS" ]] && RUN_NAME="${RUN_NAME}_filtered"
    [[ -n "$MAX_DEPTH" ]] && RUN_NAME="${RUN_NAME}_depth${MAX_DEPTH}"
    [[ -n "$EXCLUDE_COLS" ]] && RUN_NAME="${RUN_NAME}_excl"
fi

CSV_PATH="${DATASET_BASE}/decision_tree_dataset.csv"
OUTPUT_DIR="${DATASET_BASE}/${RUN_NAME}"

echo "========================================================"
echo "  Decision Tree Pipeline"
echo "========================================================"
echo "  Data folder     : $DATA_FOLDER"
echo "  Window          : [$WIN_LO, $WIN_HI)"
echo "  Stim range      : [$STIM_MIN, $STIM_MAX)"
echo "  Peak threshold  : $PEAK_THRESHOLD"
echo "  Sessions        : ${SESSIONS:-all}"
echo "  Max tree depth  : ${MAX_DEPTH:-uncapped}"
echo "  Cross-validation: $([ $NO_CV -eq 1 ] && echo 'disabled' || echo 'enabled')"
echo "  Excluded cols   : ${EXCLUDE_COLS:-none}"
echo "  Run name        : $RUN_NAME"
echo "  Output dir      : $OUTPUT_DIR"
echo "========================================================"
echo ""

# ── Step 1: Build dataset ────────────────────────────────────────────────────
echo "[ 1/2 ] Building dataset → $CSV_PATH"

BUILD_CMD="python build_decision_tree_dataset.py \
    --data_folder \"$DATA_FOLDER\" \
    --window $WIN_LO $WIN_HI \
    --stim_min $STIM_MIN \
    --stim_max_exclusive $STIM_MAX \
    --peak_threshold $PEAK_THRESHOLD \
    --output \"$CSV_PATH\""

[[ -n "$SESSIONS" ]] && BUILD_CMD="$BUILD_CMD --sessions $SESSIONS"

eval $BUILD_CMD
echo ""

# ── Step 2: Run decision tree ────────────────────────────────────────────────
echo "[ 2/2 ] Running decision tree → $OUTPUT_DIR"

TREE_CMD="python run_decision_tree.py \
    --csv \"$CSV_PATH\" \
    --output_dir \"$OUTPUT_DIR\""

[[ -n "$MAX_DEPTH"    ]] && TREE_CMD="$TREE_CMD --max_depth $MAX_DEPTH"
[[ $NO_CV -eq 1       ]] && TREE_CMD="$TREE_CMD --no_cv"
[[ -n "$EXCLUDE_COLS" ]] && TREE_CMD="$TREE_CMD --exclude_cols $EXCLUDE_COLS"

eval $TREE_CMD
echo ""

echo "Done. Results in: $OUTPUT_DIR"
