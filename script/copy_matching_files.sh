#!/usr/bin/env bash

# ============================================================
# copy_matching_files.sh
#
# Recurses through a source directory and copies files matching
# a pattern (from full_reach folders only) to a destination,
# preserving the top-level sub-directory (e.g. 210421_results/).
#
# Usage:
#   ./copy_matching_files.sh <source_dir> <dest_dir> <pattern>
#
# Example:
#   ./copy_matching_files.sh \
#     projectDir/AbigailData/energy_over_time/energy_decomp_1 \
#     ./session_visual_results/energy_decomp_1 \
#     "*.png"
# ============================================================

# --- Argument validation ---
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <source_dir> <dest_dir> <pattern>"
    exit 1
fi

SOURCE_DIR="${1%/}"
DEST_DIR="${2%/}"
PATTERN="$3"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

echo "================================================"
echo "  Source      : $SOURCE_DIR"
echo "  Destination : $DEST_DIR"
echo "  Pattern     : $PATTERN"
echo "================================================"

file_count=0
skip_count=0

while IFS= read -r filepath; do
    [[ -z "$filepath" ]] && continue

    filename="$(basename "$filepath")"
    rel_path="${filepath#"$SOURCE_DIR"/}"
    top_level_dir="${rel_path%%/*}"

    target_dir="$DEST_DIR/$top_level_dir"
    target_file="$target_dir/$filename"

    mkdir -p "$target_dir"

    if [[ -e "$target_file" ]]; then
        echo "  [SKIP] '$filename' already exists in '$target_dir'"
        ((skip_count++))
        continue
    fi

    cp "$filepath" "$target_file"
    echo "  [COPY] $rel_path  →  $top_level_dir/$filename"
    ((file_count++))

done < <(find "$SOURCE_DIR" -type d -name "full_reach" -exec find {} -maxdepth 1 -type f -name "$PATTERN" \;)

echo "================================================"
echo "  Done: $file_count file(s) copied, $skip_count skipped."
echo "================================================"