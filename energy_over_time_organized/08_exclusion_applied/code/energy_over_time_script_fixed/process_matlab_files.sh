#!/bin/bash

# Check if directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory to process
DIR=$1

# Default window size for firing rate calculation
WINDOW_SIZE=10

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist"
    exit 1
fi

# Define the reach phases with their truncation indexes and directory suffixes
# Format: "low_idx high_idx suffix description"
REACH_PHASES=(
    "100 350 begin_reach 'Beginning of reach'"
    "350 550 mid_reach 'Middle of reach'"
    "550 700 post_reach 'Post reach'"
    "100 700 full_reach 'Full reach'"
)

# Loop over all .mat files in the directory
for file in "$DIR"/*.mat; do
    # Check if file exists (in case no .mat files are found)
    if [ -f "$file" ]; then
        # Get the filename without path and extension
        filename=$(basename "$file" .mat)
        
        # Create a main folder for this file's outputs
        base_output_folder="${DIR}/${filename}_results"
        mkdir -p "$base_output_folder"
        
        echo "Processing file: $file"
        
        # Initialize combined CSV for critical values
        combined_csv="${base_output_folder}/critical_values.csv"
        echo "Phase,Critical_Temperature,Critical_Energy" > "$combined_csv"
        
        # Process each phase of the reach
        for phase in "${REACH_PHASES[@]}"; do
            # Split the phase info
            read -r low_idx high_idx suffix description <<< "$phase"
            
            # Create phase-specific output directory
            phase_output_dir="${base_output_folder}/${suffix}"
            mkdir -p "$phase_output_dir"
            
            echo "  Processing $description (indexes $low_idx-$high_idx)"
            
            # Run the Python script with the appropriate parameters
            python main.py \
                --matlab_file "$file" \
                --output_dir "$phase_output_dir" \
                --bin_size 1 \
                --sample_size 10000 \
                --n_cpus 8 \
                --max_iter 75 \
                --eta 1e-3 \
                --temp_min 0.1 \
                --temp_max 2.0 \
                --temp_step 0.05 \
                --metropolis_samples 1000000 \
                --truncate_idx_l "$low_idx" \
                --truncate_idx "$high_idx" \
                --confidence 0.8 \
                --firing_rate_window "$WINDOW_SIZE"
            
            # Copy the phase-specific critical values to the combined CSV (without header)
            # This replaces the old code that was causing duplication
            # phase_csv="${phase_output_dir}/critical_values.csv"
            # if [ -f "$phase_csv" ]; then
            #     # Extract just the phase name from the suffix
            #     phase_name="$suffix"
                
            #     # Get the values from the phase CSV (skipping header)
            #     values=$(tail -n +2 "$phase_csv")
                
            #     # Only add to combined CSV if values exist
            #     if [ -n "$values" ]; then
            #         # Add to combined CSV with correct phase name
            #         echo "$phase_name,$(echo "$values" | cut -d',' -f2-)" >> "$combined_csv"
            #     fi
            # fi
        done
        
        echo "Completed processing for $file"
        echo "Results saved to $base_output_folder"
        echo "Critical values combined in $combined_csv"
        echo "-------------------------"
    fi
done

echo "All files processed"