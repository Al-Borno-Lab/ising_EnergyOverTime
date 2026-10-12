# Wells Data Utilities

This directory contains utilities for processing and analyzing neural and behavioral data from the Wells lab experiments.

## Data Structure

### Input Files

1. `df_neural.json`
   - Contains neural recording data
   - Key fields:
     - `overall_rate`: Neural firing rates for each unit
     - `waveform_duration`: Waveform duration measurements
     - `waveform_PTratio`: Peak-to-trough ratio measurements
     - `waveform_repolarizationslope`: Repolarization slope measurements
     - `waveform_class`: Waveform classification
     - `depth`: Depth measurements for each unit
     - `layer`: Layer information for each unit
     - `times`: Spike times for each unit
     - `mouse`: Mouse ID for each unit

2. `df_reaches.json`
   - Contains reach-related behavioral data
   - Key fields:
     - `xpos_toEnd`: X positions during reach
     - `ypos_toEnd`: Y positions during reach
     - `reachMax_ephys`: Time of reach maximum in ephys time
     - `reachMax_ind`: Index of reach maximum
     - `mouse`: Mouse ID for each reach

### Output Structure (aligned_data.json)

The aligned data is organized in a hierarchical structure:

```json
{
    "behavior": {
        "data": [
            // First level: Mice
            [
                // Second level: Reaches for each mouse
                [
                    // Third level: Position data for each reach
                    [x1, y1], [x2, y2], ...
                ],
                ...
            ],
            ...
        ],
        "time": [
            // First level: Mice
            [
                // Second level: Time points for each reach
                [t1, t2, ...],
                ...
            ],
            ...
        ]
    },
    "neural": {
        "data": [
            // First level: Mice
            [
                // Second level: Reaches
                [
                    // Third level: Neurons
                    [spike1, spike2, ...],
                    ...
                ],
                ...
            ],
            ...
        ],
        "time": [
            // First level: Mice
            [
                // Second level: Time points for each reach
                [t1, t2, ...],
                ...
            ],
            ...
        ],
        "metadata": [
            // First level: Mice
            [
                // Second level: Neurons
                {
                    "depth": depth_value,
                    "layer": layer_value
                },
                ...
            ],
            ...
        ]
    }
}
```

## Scripts

1. `alignment.py`
   - Processes reach and neural data
   - Aligns behavioral and neural data in time
   - Organizes data by reach

2. `alignment-conversion.py`
   - Converts aligned data into a specific JSON structure
   - Includes both behavior and neural arrays
   - Adds metadata (depth and layer) for each neuron
   - Organizes data by mouse and reach

3. `check_json_structure.py`
   - Analyzes and displays the structure of JSON files
   - Useful for verifying data format and contents

4. `summarize.py`
   - Provides summary statistics and information about the data

## Data Processing

The data processing pipeline:
1. Loads raw neural and reach data
2. Aligns behavior and neural data in time
3. Organizes data by mouse and reach
4. Adds metadata for each neuron
5. Saves the processed data in a structured JSON format

## Notes

- Time vectors are aligned relative to reach maximum
- Neural data is sampled at 1ms resolution
- Behavior data is sampled at 150Hz
- All data is organized hierarchically by mouse, then by reach 