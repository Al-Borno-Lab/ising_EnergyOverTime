#!/bin/bash
#SBATCH --job-name=ising_master
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --output=ising_master_%j.log

set -euo pipefail

# Path to your singularity container
CONTAINER="~/projectDir/singularity-env/inverse-ising-arm-2.sif"

# Directory containing main.py / arbitration_many.py (for follow-up job).
# When you run sbatch from the energy_over_time_script folder, SLURM_SUBMIT_DIR is set correctly.
_resolve_energy_script_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "${SLURM_SUBMIT_DIR}/arbitration_many.py" ]]; then
        (cd "${SLURM_SUBMIT_DIR}" && pwd)
        return
    fi
    local _script="${BASH_SOURCE[0]}"
    if [[ "${_script}" != /* ]]; then
        _script="${PWD}/${_script}"
    fi
    if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
        _script="$(readlink -f "${_script}")"
    fi
    (cd "$(dirname "${_script}")" && pwd)
}

ENERGY_SCRIPT_DIR="$(_resolve_energy_script_dir)"

# Check if directory and number of repetitions are provided
if [ $# -lt 3 ]; then
    echo "Usage: sbatch process_matlab_file_manyTimes_sbatch.sh <directory> <number_of_repetitions> <output_dir>"
    echo "  Submit from energy_over_time_script (or a directory that contains arbitration_many.py) so the arbitration step can find the Python tree."
    exit 1
fi

# Directory to process and number of repetitions
DIR=$1
NUM_REPETITIONS=$2
OUTPUT_DIR=$3

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# Expand ~ in container path for generated job scripts
CONTAINER_EXPANDED="${CONTAINER/#\~/${HOME}}"

# Define window size for firing rate calculation
WINDOW_SIZE=1

# Collect ising_task job IDs for Slurm afterok dependency
JOB_IDS=()

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist"
    exit 1
fi

# Validate NUM_REPETITIONS is a positive integer
if ! [[ "$NUM_REPETITIONS" =~ ^[0-9]+$ ]] || [ "$NUM_REPETITIONS" -lt 1 ]; then
    echo "Error: Number of repetitions must be a positive integer"
    exit 1
fi

# Arbitration rep range must include all repetition folders (rep 1 .. NUM_REPETITIONS)
REP_END_EXCLUSIVE=$((NUM_REPETITIONS + 1))

# Define the reach phases with their truncation indexes and directory suffixes
# Format: "low_idx high_idx suffix description"
REACH_PHASES=(
    "100 350 begin_reach 'Beginning of reach'"
    "350 500 mid_reach 'Middle of reach'"
    "500 800 post_reach 'Post reach'"
    "100 800 full_reach 'Full reach'"
)

# Create a template for the individual job script
cat > job_template.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=ising_task
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=./logs_window_4/ising_task_%j.log

# Arguments passed to this script
MATLAB_FILE=$1
OUTPUT_DIR=$2
LOW_IDX=$3
HIGH_IDX=$4
WINDOW_SIZE=$5

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script inside the singularity container
singularity exec CONTAINER_PATH /entrypoint.sh python main.py \
    --matlab_file "$MATLAB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --bin_size 1 \
    --sample_size 100000 \
    --n_cpus 64 \
    --max_iter 200 \
    --eta 0.05 \
    --temp_min 0.1 \
    --temp_max 2.0 \
    --temp_step 0.05 \
    --metropolis_samples 100000 \
    --truncate_idx_l "$LOW_IDX" \
    --truncate_idx "$HIGH_IDX" \
    --confidence 0.8 \
    --firing_rate_window "$WINDOW_SIZE"

echo "Task completed for $OUTPUT_DIR"
EOF

# Replace the container path in the template
sed -i "s|CONTAINER_PATH|${CONTAINER_EXPANDED}|g" job_template.sh
chmod +x job_template.sh

# Loop over all .mat files in the directory
for file in "$DIR"/*.mat; do
    # Check if file exists (in case no .mat files are found)
    if [ -f "$file" ]; then
        # Get the filename without path and extension
        filename=$(basename "$file" .mat)
        
        # Extract experiment name (assuming it's the first part of the filename before any special characters)
        experiment_name=$(echo "$filename" | cut -d'_' -f1)
        
        # Create a main folder for this experiment's outputs
        base_output_folder="${OUTPUT_DIR}/${experiment_name}_results"
        mkdir -p "$base_output_folder"
        
        echo "Processing file: $file"
        echo "Experiment name: $experiment_name"
        
        # Process each phase of the reach
        for phase in "${REACH_PHASES[@]}"; do
            # Split the phase info
            read -r low_idx high_idx suffix description <<< "$phase"
            
            echo "  Processing $description (indexes $low_idx-$high_idx)"
            
            # Submit N repetition jobs for this phase
            for (( rep=1; rep<=$NUM_REPETITIONS; rep++ )); do
                # Create repetition-specific output directory with experiment name
                rep_output_dir="${base_output_folder}/${experiment_name}_rep${rep}/${suffix}"
                
                echo "    Submitting job for repetition $rep of $NUM_REPETITIONS"
                
                jid=$(sbatch --parsable job_template.sh "$file" "$rep_output_dir" "$low_idx" "$high_idx" "$WINDOW_SIZE")
                jid="${jid%%;*}"
                if [[ -z "${jid}" ]] || ! [[ "${jid}" =~ ^[0-9]+$ ]]; then
                    echo "Error: sbatch failed or returned unexpected id: ${jid}" >&2
                    exit 1
                fi
                JOB_IDS+=("${jid}")
                
                # Add a small delay to avoid overwhelming the scheduler
                sleep 0.5
            done
        done
        
        echo "All jobs submitted for $file"
        echo "Results will be saved to $base_output_folder"
        echo "-------------------------"
    fi
done

# Clean up the template
rm job_template.sh

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
    echo "No .mat jobs were submitted; skipping arbitration follow-up."
    echo "All jobs have been submitted (none)."
    exit 0
fi

# Build afterok dependency: jobid1:jobid2:...
DEP_STRING=$(IFS=:; echo "${JOB_IDS[*]}")
MASTER_TAG="${SLURM_JOB_ID:-$$}"

FOLLOWUP_SH="${OUTPUT_DIR}/slurm_arbitration_followup_${MASTER_TAG}.sh"

cat > "${FOLLOWUP_SH}" << FOLLOWUP_EOF
#!/bin/bash
#SBATCH --job-name=ising_arbitration
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=${OUTPUT_DIR}/ising_arbitration_%j.log

set -euo pipefail
cd "${ENERGY_SCRIPT_DIR}"

echo "Running arbitration_many.py on ${OUTPUT_DIR}"
echo "Plots and CSVs -> ${OUTPUT_DIR}/arbitration/"

singularity exec "${CONTAINER_EXPANDED}" /entrypoint.sh python arbitration_many.py \\
    --data_folder "${OUTPUT_DIR}" \\
    --output_base "${OUTPUT_DIR}/arbitration" \\
    --rep_start 1 \\
    --rep_end_exclusive ${REP_END_EXCLUSIVE} \\
    --window 390 410 \\
    --quiet_find

echo "Arbitration job finished."
FOLLOWUP_EOF

chmod +x "${FOLLOWUP_SH}"

ARBIT_JID=$(sbatch --parsable --dependency=afterok:"${DEP_STRING}" "${FOLLOWUP_SH}")
ARBIT_JID="${ARBIT_JID%%;*}"

echo "Submitted ${#JOB_IDS[@]} ising_task job(s)."
echo "Dependency chain: afterok:${DEP_STRING}"
echo "Arbitration follow-up job ID: ${ARBIT_JID}"
echo "Follow-up script (for reference): ${FOLLOWUP_SH}"
echo "Arbitration Slurm log: ${OUTPUT_DIR}/ising_arbitration_<jobid>.log"
echo "All jobs have been submitted"
