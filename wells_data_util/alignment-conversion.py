import json
import numpy as np
from tqdm import tqdm

def load_data():
    # Load reach data
    with open('data/wells_data/df_reaches.json', 'r') as f:
        reach_data = json.load(f)
    
    # Load neural data
    with open('data/wells_data/df_neural.json', 'r') as f:
        neural_data = json.load(f)
    
    return reach_data, neural_data

def process_behavior_data(reach_data):
    behavior = {}
    behavior_time = {}
    
    # Get unique mouse IDs from reach data
    mouse_ids = set(reach_data['mouse'].values())
    
    for mouse_id in tqdm(mouse_ids, desc="Processing behavior data"):
        # Get all reaches for this mouse
        mouse_reaches = [k for k, v in reach_data['mouse'].items() if v == mouse_id]
        
        # Initialize mouse behavior arrays
        mouse_behavior = []
        mouse_behavior_time = []
        
        for reach_id in mouse_reaches:
            # Get x and y positions for this reach
            x_pos = reach_data['xpos_toEnd'][reach_id]
            y_pos = reach_data['ypos_toEnd'][reach_id]
            
            # Get the reach maximum time from ephys data
            reach_max_ephys = reach_data['reachMax_ephys'][reach_id]
            reach_max_idx = reach_data['reachMax_ind'][reach_id]
            
            # Calculate time points relative to reach maximum
            # Assuming 150Hz sampling rate for behavior data
            time_points = np.arange(-reach_max_idx/150.0, (len(x_pos)-reach_max_idx)/150.0, 1/150.0)
            
            # Combine x, y positions and time into a single array
            positions = list(zip(x_pos, y_pos, time_points))
            mouse_behavior.append(positions)
            mouse_behavior_time.append(time_points.tolist())
        
        behavior[str(mouse_id)] = mouse_behavior
        behavior_time[str(mouse_id)] = mouse_behavior_time
    
    return behavior, behavior_time

def process_neural_data(neural_data, reach_data):
    neural = {}
    neural_time = {}
    neural_metadata = {}
    
    # Get unique mouse IDs from neural data
    mouse_ids = set(neural_data['mouse'].values())
    
    for mouse_id in tqdm(mouse_ids, desc="Processing neural data"):
        # Get all neurons for this mouse
        mouse_neurons = [k for k, v in neural_data['mouse'].items() if v == mouse_id]
        
        # Initialize mouse neural arrays
        mouse_neural = []
        mouse_neural_time = []
        mouse_metadata = []
        
        # Get all reaches for this mouse
        mouse_reaches = [k for k, v in reach_data['mouse'].items() if v == mouse_id]
        
        for reach_id in mouse_reaches:
            # Get the reach maximum time from ephys data
            reach_max_ephys = reach_data['reachMax_ephys'][reach_id]
            
            # Define time window around reach maximum
            time_before = 0.6  # seconds before reach max
            time_after = 0.4   # seconds after reach max
            
            # Create time vector for this reach
            time_points = np.arange(-time_before, time_after, 0.001)  # 1ms resolution
            mouse_neural_time.append(time_points.tolist())
            
            # Process each neuron for this reach
            reach_neural = []
            for neuron_id in mouse_neurons:
                spike_times = neural_data['times'][neuron_id]
                
                # Create sparse array for this time window
                sparse_array = np.zeros(len(time_points))
                
                # Convert spike times to indices and set to 1
                for spike_time in spike_times:
                    # Calculate time relative to reach maximum
                    relative_time = spike_time - reach_max_ephys
                    idx = int((relative_time + time_before) * 1000)  # Convert to ms index
                    if 0 <= idx < len(sparse_array):
                        sparse_array[idx] = 1
                
                reach_neural.append(sparse_array.tolist())
            
            mouse_neural.append(reach_neural)
        
        # Store metadata for each neuron in this mouse
        for neuron_id in mouse_neurons:
            neuron_metadata = {
                'depth': neural_data['depth'][neuron_id],
                'layer': neural_data['layer'][neuron_id],
                'waveform_duration': neural_data['waveform_duration'][neuron_id],
                'waveform_PTratio': neural_data['waveform_PTratio'][neuron_id],
                'waveform_repolarizationslope': neural_data['waveform_repolarizationslope'][neuron_id],
                'waveform_class': neural_data['waveform_class'][neuron_id]
            }
            mouse_metadata.append(neuron_metadata)
        
        neural[str(mouse_id)] = mouse_neural
        neural_time[str(mouse_id)] = mouse_neural_time
        neural_metadata[str(mouse_id)] = mouse_metadata
    
    return neural, neural_time, neural_metadata

def main():
    # Load data
    reach_data, neural_data = load_data()
    
    # Process behavior data
    behavior, behavior_time = process_behavior_data(reach_data)
    
    # Process neural data
    neural, neural_time, neural_metadata = process_neural_data(neural_data, reach_data)
    
    # Create output structure
    output = {
        "behavior": {
            "data": behavior,
            "time": behavior_time
        },
        "neural": {
            "data": neural,
            "time": neural_time,
            "metadata": neural_metadata
        }
    }
    
    # Save to file
    with open('aligned_data.json', 'w') as f:
        json.dump(output, f)
    
    print("Data processing complete. Output saved to aligned_data.json")

if __name__ == "__main__":
    main()
