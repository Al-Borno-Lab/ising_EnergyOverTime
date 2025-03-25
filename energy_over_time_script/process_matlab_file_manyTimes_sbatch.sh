#!/bin/bash
#SBATCH --job-name=ising_master
#SBATCH --partition=math-alderaan
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --output=ising_master_%j.log

# Path to your singularity container
CONTAINER="~/projectDir/singularity-env/inverseIsing.sif"

# Check if directory and number of repetitions are provided
if [ $# -lt 2 ]; then
    echo "Usage: sbatch run_ising_master.slurm <directory> <number_of_repetitions>"
    exit 1
fi

# Directory to process and number of repetitions
DIR=$1
NUM_REPETITIONS=$2

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

# Define the reach phases with their truncation indexes and directory suffixes
# Format: "low_idx high_idx suffix description"
REACH_PHASES=(
    "100 300 begin_reach 'Beginning of reach'"
    "300 500 mid_reach 'Middle of reach'"
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=ising_task_%j.log

# Arguments passed to this script
MATLAB_FILE=$1
OUTPUT_DIR=$2
LOW_IDX=$3
HIGH_IDX=$4
WINDOW_SIZE=$5

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script inside the singularity container
singularity exec CONTAINER_PATH python main.py \
    --matlab_file "$MATLAB_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --bin_size 1 \
    --sample_size 10000 \
    --n_cpus 8 \
    --max_iter 75 \
    --eta 1e-3 \
    --temp_min 0.1 \
    --temp_max 2.0 \
    --temp_step 0.05 \
    --metropolis_samples 1000000 \
    --truncate_idx_l "$LOW_IDX" \
    --truncate_idx "$HIGH_IDX" \
    --confidence 0.8 \
    --firing_rate_window "$WINDOW_SIZE"

echo "Task completed for $OUTPUT_DIR"
EOF

# Replace the container path in the template
sed -i "s|CONTAINER_PATH|$CONTAINER|g" job_template.sh
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
        base_output_folder="${DIR}/${experiment_name}_results"
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
                
                # Submit the job
                sbatch job_template.sh "$file" "$rep_output_dir" "$low_idx" "$high_idx" "$WINDOW_SIZE"
                
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

echo "All jobs have been submitted"