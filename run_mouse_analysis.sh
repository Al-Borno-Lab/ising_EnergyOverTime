#!/bin/bash

# Array of mouse IDs to process
MOUSE_IDS=(103 104 105 106 107)  # replace with your actual mouse IDs

# Data file path
WELLS_FILE="data/aligned_data.json"

# Output log file
LOG_FILE="mouse_analysis_log.txt"

echo "Starting analysis for all mice at $(date)" > $LOG_FILE

# Loop through each mouse ID and run the analysis
for MOUSE_ID in "${MOUSE_IDS[@]}"; do
    echo "Processing mouse ID: $MOUSE_ID" | tee -a $LOG_FILE
    
    # Run the Python script with the current mouse ID
    python main-wells.py --wells_file "$WELLS_FILE" --mouse_id "$MOUSE_ID"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed mouse ID: $MOUSE_ID" | tee -a $LOG_FILE
    else
        echo "✗ Error processing mouse ID: $MOUSE_ID" | tee -a $LOG_FILE
    fi
    
    echo "------------------------" | tee -a $LOG_FILE
done

echo "Analysis completed for all mice at $(date)" | tee -a $LOG_FILE

# Make the script executable automatically
chmod +x run_mouse_analysis.sh 