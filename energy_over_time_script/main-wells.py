#!/usr/bin/env python
# coding: utf-8

"""
Main script for analyzing neural data from Wells lab experiments using the Ising model.

This script processes neural and behavioral data from Wells lab experiments to:
1. Load and preprocess neural data from aligned_data.json
2. Filter neurons by specified layers
3. Fit an Ising model to the spike data
4. Analyze phase transitions in the model
5. Calculate energy for neural states
6. Generate visualizations of the results
7. Output data to CSV files for further analysis

Usage:
    python main-wells.py --mouse_id MOUSE_ID --layers LAYER1 LAYER2 [--options]
"""

import os
import argparse
import numpy as np
import time
import pandas as pd
import json
from os.path import basename, splitext

# Import custom modules
from utils import preprocessingSpikes, calculate_statistics_with_ci
from model import fit_ising_model, phase_transition_analysis, calculate_energy_for_spike_data
from analysis import (analyze_wells_data, calculate_statistics_across_reaches, 
                     identify_reach_transition_points, create_energy_spline)
from visualization import (create_output_directory, plot_phase_transition, 
                         plot_energy_across_time, plot_transition_points, 
                         plot_energy_histogram, plot_model_quality)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Analysis of neural data from Wells lab experiments using the Ising model.')
    
    parser.add_argument('--mouse_id', type=str, required=True, 
                        help='Mouse ID to analyze')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='List of layers to analyze (e.g., "L2/3" "L4" "L5"). If not specified, all layers will be used.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: derived from mouse ID and layers)')
    parser.add_argument('--bin_size', type=int, default=1,
                        help='Bin size for preprocessing spike data (default: 1)')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='Sample size for the Ising model fitting (default: 10000)')
    parser.add_argument('--n_cpus', type=int, default=8,
                        help='Number of CPUs to use for model fitting (default: 8)')
    parser.add_argument('--max_iter', type=int, default=75,
                        help='Maximum number of iterations for model fitting (default: 75)')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='Learning rate for model fitting (default: 1e-3)')
    parser.add_argument('--temp_min', type=float, default=0.1,
                        help='Minimum temperature for phase transition analysis (default: 0.1)')
    parser.add_argument('--temp_max', type=float, default=2.0,
                        help='Maximum temperature for phase transition analysis (default: 2.0)')
    parser.add_argument('--temp_step', type=float, default=0.05,
                        help='Temperature step for phase transition analysis (default: 0.05)')
    parser.add_argument('--metropolis_samples', type=int, default=1000000,
                        help='Number of samples for Metropolis sampling (default: 1000000)')
    parser.add_argument('--confidence', type=float, default=0.8,
                        help='Confidence level for statistical intervals (default: 0.8)')
    parser.add_argument('--skip_phase_analysis', action='store_true',
                        help='Skip phase transition analysis (useful for quick testing)')
    parser.add_argument('--firing_rate_window', type=int, default=10,
                        help='Window size for time-dependent firing rate calculation (default: 10)')
    
    return parser.parse_args()

def load_wells_data(mouse_id, layers):
    """
    Load and preprocess Wells lab data for a specific mouse and layers.
    
    Parameters:
    -----------
    mouse_id : str
        Mouse ID to analyze
    layers : list of str
        List of layers to analyze
        
    Returns:
    --------
    tuple
        (neural_data, behavior_data, neural_time, behavior_time)
    """
    # Load aligned data
    with open('aligned_data.json', 'r') as f:
        data = json.load(f)
    
    # Find mouse index
    mouse_idx = None
    for i, mouse_data in enumerate(data['neural']['metadata']):
        if mouse_data[0]['mouse'] == mouse_id:
            mouse_idx = i
            break
    
    if mouse_idx is None:
        raise ValueError(f"Mouse ID {mouse_id} not found in data")
    
    # If layers is None, use all unique layers for this mouse
    if layers is None:
        layers = list(set(neuron['layer'] for neuron in data['neural']['metadata'][mouse_idx]))
    
    # Filter neurons by layers
    layer_neurons = []
    for i, neuron in enumerate(data['neural']['metadata'][mouse_idx]):
        if neuron['layer'] in layers:
            layer_neurons.append(i)
    
    if not layer_neurons:
        raise ValueError(f"No neurons found in layers {layers} for mouse {mouse_id}")
    
    # Extract neural data for selected neurons
    neural_data = []
    for reach in data['neural']['data'][mouse_idx]:
        reach_data = []
        for timepoint in reach:
            reach_data.append([timepoint[i] for i in layer_neurons])
        neural_data.append(reach_data)
    
    # Extract behavior data
    behavior_data = data['behavior']['data'][mouse_id]
    
    # Extract time vectors
    neural_time = data['neural']['time'][mouse_idx]
    behavior_time = data['behavior']['time'][mouse_id]
    
    return np.array(neural_data), np.array(behavior_data), np.array(neural_time), np.array(behavior_time)

def run_analysis(args):
    """
    Run the full analysis pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    None
    """
    # Set output directory
    if args.output_dir is None:
        layer_str = "_".join(args.layers)
        args.output_dir = f"mouse_{args.mouse_id}_{layer_str}_results"
    
    output_dir = create_output_directory(args.output_dir)
    
    # Load and preprocess data
    print(f"Loading data for mouse {args.mouse_id}, layers {args.layers}...")
    neural_data, behavior_data, neural_time, behavior_time = load_wells_data(args.mouse_id, args.layers)
    
    # Preprocess spike data for each reach
    print("Preprocessing spike data for model fitting...")
    all_reaches_data = []
    for reach in neural_data:
        bin_cat = np.vstack(reach)
        bin_cat_p = preprocessingSpikes(bin_cat, args.bin_size)
        bin_cat_p = 2 * bin_cat_p - 1  # Convert to {-1, 1} representation
        all_reaches_data.append(bin_cat_p)
    
    # Combine all reaches for model fitting
    combined_data = np.vstack(all_reaches_data)
    N = combined_data.shape[1]
    print(f"Number of neurons: {N}")
    print(f"Number of reaches: {len(neural_data)}")
    
    # Save preprocessing info to CSV
    preprocessing_info = {
        'Mouse_ID': args.mouse_id,
        'Layers': ", ".join(args.layers),
        'Number_of_Neurons': N,
        'Number_of_Reaches': len(neural_data),
        'Bin_Size': args.bin_size
    }
    pd.DataFrame([preprocessing_info]).to_csv(os.path.join(output_dir, "preprocessing_info.csv"), index=False)
    
    # Fit Ising model
    print("Fitting Ising model...")
    multipliers, solver = fit_ising_model(
        combined_data,
        sample_size=args.sample_size,
        n_cpus=args.n_cpus,
        max_iter=args.max_iter,
        eta=args.eta
    )
    
    # Save model parameters to CSV
    h_params = multipliers[:N]
    J_params = multipliers[N:]
    
    model_params_df = pd.DataFrame({
        'h_values': h_params
    })
    model_params_df.to_csv(os.path.join(output_dir, "h_parameters.csv"), index=False)
    
    # Create a dataframe for J parameters (coupling matrix)
    J_dict = {}
    k = 0
    for i in range(N-1):
        for j in range(i+1, N):
            J_dict[f'J_{i+1}_{j+1}'] = J_params[k]
            k += 1
    
    J_params_df = pd.DataFrame([J_dict])
    J_params_df.to_csv(os.path.join(output_dir, "J_parameters.csv"), index=False)
    
    # Evaluate model quality
    print("Evaluating model quality...")
    plot_model_quality(combined_data, solver.model.sample, output_dir)
    
    # Phase transition analysis
    if not args.skip_phase_analysis:
        print("Analyzing phase transitions...")
        temp_range = np.arange(args.temp_min, args.temp_max, args.temp_step)
        pt_results = phase_transition_analysis(
            multipliers, 
            N, 
            temp_range=temp_range,
            samples=args.metropolis_samples
        )
        
        # Plot phase transition results
        energy_spline = plot_phase_transition(pt_results, output_dir)
        
        critical_temp = pt_results['critical_temp']
        critical_energy = pt_results['critical_energy']
        
        # Save critical values to file
        with open(os.path.join(output_dir, "critical_values.txt"), "w") as f:
            f.write(f"Mouse ID: {args.mouse_id}\n")
            f.write(f"Layers: {', '.join(args.layers)}\n")
            f.write(f"Number of Neurons: {N}\n")
            f.write(f"Number of Reaches: {len(neural_data)}\n")
            f.write(f"Critical Temperature: {critical_temp}\n")
            f.write(f"Critical Energy: {critical_energy}\n")
    else:
        print("Skipping phase transition analysis...")
        critical_energy = -5.0  # Placeholder value
        energy_spline = None
    
    # Analyze all reaches using Wells-specific analysis
    print("Analyzing all reaches...")
    analysis_results = analyze_wells_data(
        neural_data,
        behavior_data,
        multipliers,
        critical_energy,
        energy_temp_spline=energy_spline,
        output_dir=output_dir
    )
    
    # Calculate statistics across reaches
    print("Calculating statistics across reaches...")
    reach_statistics = calculate_statistics_across_reaches(
        analysis_results,
        confidence=args.confidence,
        output_dir=output_dir
    )
    
    # Analyze each reach individually
    print("Analyzing individual reaches...")
    for reach_idx, (reach_data, reach_behavior) in enumerate(zip(neural_data, behavior_data)):
        reach_dir = os.path.join(output_dir, f"reach_{reach_idx}")
        os.makedirs(reach_dir, exist_ok=True)
        
        # Get this reach's data from analysis results
        energy_data = analysis_results['energy_values'][reach_idx]
        x_positions = analysis_results['x_positions'][reach_idx]
        y_positions = analysis_results['y_positions'][reach_idx]
        
        # Plot energy across time for this reach
        plot_energy_across_time(
            reach_statistics, 
            critical_energy, 
            reach_dir,
            title_prefix=f"Mouse {args.mouse_id} Reach {reach_idx}",
            neural_data=[reach_data],
            window_size=args.firing_rate_window
        )
        
        # Identify transition points
        transition_points = identify_reach_transition_points(
            energy_data,
            x_positions,
            y_positions,
            threshold=0.2,
            output_dir=reach_dir,
            reach_idx=reach_idx
        )
        
        # Plot transition points
        if len(transition_points) > 0:
            plot_transition_points(
                energy_data,
                np.column_stack((x_positions, y_positions)),
                transition_points,
                reach_dir
            )
            
            print(f"Identified {len(transition_points)} transition points in reach {reach_idx}")
        
        # Plot energy histogram
        plot_energy_histogram(
            energy_data,
            critical_energy,
            reach_dir
        )
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    
    # Run the analysis
    run_analysis(args)
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
