import json
import numpy as np
from tqdm import tqdm
import concurrent.futures
import os
import sys

def load_data():
    # Load reach data
    with open('data/wells_data/df_reaches.json', 'r') as f:
        reach_data = json.load(f)
    
    # Load neural data
    with open('data/wells_data/df_neural.json', 'r') as f:
        neural_data = json.load(f)
    
    return reach_data, neural_data

def process_single_reach_behavior(args):
    reach_id, reach_data = args
    x_pos = reach_data['xpos_toEnd'][reach_id]
    y_pos = reach_data['ypos_toEnd'][reach_id]
    reach_max_ephys = reach_data['reachMax_ephys'][reach_id]
    reach_max_idx = reach_data['reachMax_ind'][reach_id]
    time_points = np.arange(-reach_max_idx/150.0, (len(x_pos)-reach_max_idx)/150.0, 1/150.0)
    positions = list(zip(x_pos, y_pos, time_points))
    return positions, time_points.tolist()

def process_behavior_data(reach_data):
    behavior = {}
    behavior_time = {}
    
    # Get unique mouse IDs from reach data
    mouse_ids = set(reach_data['mouse'].values())
    
    for mouse_id in tqdm(mouse_ids, desc="Processing behavior data"):
        # Get all reaches for this mouse
        mouse_reaches = [k for k, v in reach_data['mouse'].items() if v == mouse_id]
        
        mouse_behavior = []
        mouse_behavior_time = []
        # Parallel processing of reaches
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_single_reach_behavior, [(reach_id, reach_data) for reach_id in mouse_reaches]))
        for positions, time_points in results:
            mouse_behavior.append(positions)
            mouse_behavior_time.append(time_points)
        behavior[str(mouse_id)] = mouse_behavior
        behavior_time[str(mouse_id)] = mouse_behavior_time
    
    return behavior, behavior_time

def process_single_reach_neural(args):
    reach_id, mouse_neurons, neural_data, reach_data = args
    reach_max_ephys = reach_data['reachMax_ephys'][reach_id]
    time_before = 0.6  # seconds before reach max
    time_after = 0.4   # seconds after reach max
    time_points = np.arange(-time_before, time_after, 0.001)  # 1ms resolution
    reach_neural = []
    for neuron_id in mouse_neurons:
        spike_times = neural_data['times'][neuron_id]
        sparse_array = np.zeros(len(time_points))
        for spike_time in spike_times:
            relative_time = spike_time - reach_max_ephys
            idx = int((relative_time + time_before) * 1000)  # Convert to ms index
            if 0 <= idx < len(sparse_array):
                sparse_array[idx] = 1
        reach_neural.append(sparse_array.tolist())
    return reach_neural, time_points.tolist()

def process_neural_data(neural_data, reach_data, layer=None):
    neural = {}
    neural_time = {}
    neural_metadata = {}
    
    # Get unique mouse IDs from neural data
    mouse_ids = set(neural_data['mouse'].values())
    
    for mouse_id in tqdm(mouse_ids, desc="Processing neural data"):
        # Get all neurons for this mouse using the mouse mapping
        if layer is not None:
            mouse_neurons = [neuron_id for neuron_id, m_id in neural_data['mouse'].items() if m_id == mouse_id and neural_data['layer'][neuron_id] == layer]
        else:
            mouse_neurons = [neuron_id for neuron_id, m_id in neural_data['mouse'].items() if m_id == mouse_id]
        
        mouse_neural = []
        mouse_neural_time = []
        mouse_metadata = []
        
        # Get all reaches for this mouse
        mouse_reaches = [k for k, v in reach_data['mouse'].items() if v == mouse_id]
        
        # Parallel processing of reaches
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_single_reach_neural, [
                (reach_id, mouse_neurons, neural_data, reach_data) for reach_id in mouse_reaches
            ]))
        for reach_neural, time_points in results:
            mouse_neural.append(reach_neural)
            mouse_neural_time.append(time_points)
        
        # Store metadata for each neuron in this mouse
        for neuron_id in mouse_neurons:
            neuron_metadata = {
                'neuron_id': neuron_id,
                'layer': neural_data['layer'][neuron_id],
                'spike_times': neural_data['times'][neuron_id]
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
    
    # Check for layer argument
    layer = None
    if len(sys.argv) > 1:
        layer = sys.argv[1] if sys.argv[1].lower() != 'none' else None
    
    # Process neural data
    neural, neural_time, neural_metadata = process_neural_data(neural_data, reach_data, layer=layer)
    
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
