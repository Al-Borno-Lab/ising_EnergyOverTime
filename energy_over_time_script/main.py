#!/usr/bin/env python
# coding: utf-8

"""
Main script for analyzing Purkinje neuron spike configurations using the Ising model.

This script ties together the functionality from the other modules to:
1. Load and preprocess neural data from MATLAB files
2. Fit an Ising model to the spike data
3. Analyze phase transitions in the model
4. Calculate energy for neural states
5. Generate visualizations of the results
6. Output data to CSV files for further analysis

Usage:
    python main.py --matlab_file path/to/file.mat [--options]
"""

import os
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from os.path import basename, splitext

# Import custom modules
from utils import load_and_preprocess_data, preprocessingSpikes, calculate_statistics_with_ci
from model import fit_ising_model, phase_transition_analysis, calculate_energy_for_spike_data
from analysis import create_energy_spline, analyze_neural_stimuli, calculate_statistics_across_trials, identify_transition_points
from visualization import create_output_directory, plot_phase_transition, plot_energy_across_time, plot_transition_points, plot_energy_histogram, plot_model_quality

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Analysis of Purkinje neuron spike configurations using the Ising model.')
    
    parser.add_argument('--matlab_file', type=str, required=True, 
                        help='Path to the MATLAB .mat file containing the neural data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: derived from MATLAB filename)')
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
    parser.add_argument('--truncate_idx_l', type=int, default=100,
                        help='Lower truncation index for data preprocessing (default: 100)')
    parser.add_argument('--truncate_idx', type=int, default=800,
                        help='Upper truncation index for data preprocessing (default: 800)')
    parser.add_argument('--confidence', type=float, default=0.8,
                        help='Confidence level for statistical intervals (default: 0.8)')
    parser.add_argument('--skip_phase_analysis', action='store_true',
                        help='Skip phase transition analysis (useful for quick testing)')
    
    return parser.parse_args()

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
        # Create output directory based on MATLAB filename
        matlab_basename = splitext(basename(args.matlab_file))[0]
        args.output_dir = matlab_basename
    
    output_dir = create_output_directory(args.output_dir)
    
    # Load and preprocess data
    print(f"Loading data from {args.matlab_file}...")
    neural_stim, continous_stim = load_and_preprocess_data(
        args.matlab_file, 
        truncate_idx_l=args.truncate_idx_l, 
        truncate_idx=args.truncate_idx
    )
    
    # Extract and preprocess first stimulus for model fitting
    print("Preprocessing spike data for model fitting...")
    stim_0 = neural_stim[0]
    bin_cat = np.vstack(stim_0)
    
    bin_cat_p = preprocessingSpikes(bin_cat, args.bin_size)
    bin_cat_p = 2 * bin_cat_p - 1  # Convert to {-1, 1} representation
    
    N = bin_cat_p.shape[1]
    print(f"Number of neurons: {N}")
    
    # Save preprocessing info to CSV
    preprocessing_info = {
        'Number_of_Neurons': N,
        'Bin_Size': args.bin_size,
        'Truncate_Index_Lower': args.truncate_idx_l,
        'Truncate_Index_Upper': args.truncate_idx,
        'Reach_Phase': os.path.basename(output_dir)
    }
    pd.DataFrame([preprocessing_info]).to_csv(os.path.join(output_dir, "preprocessing_info.csv"), index=False)
    
    # Fit Ising model
    print("Fitting Ising model...")
    multipliers, solver = fit_ising_model(
        bin_cat_p,
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
    plot_model_quality(bin_cat_p, solver.model.sample, output_dir)
    
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
            f.write(f"Number of Neurons: {N}\n")
            f.write(f"Critical Temperature: {critical_temp}\n")
            f.write(f"Critical Energy: {critical_energy}\n")
    else:
        print("Skipping phase transition analysis...")
        # Use default critical energy if skipping analysis
        critical_energy = -5.0  # This is a placeholder value
        energy_spline = None
    
    # Analyze neural stimuli
    print("Analyzing neural stimuli...")
    analysis_results = analyze_neural_stimuli(
        neural_stim, 
        continous_stim, 
        multipliers, 
        critical_energy,
        energy_temp_spline=energy_spline
    )
    
    # Calculate statistics across trials
    print("Calculating statistics across trials...")
    trial_statistics = calculate_statistics_across_trials(
        analysis_results,
        confidence=args.confidence
    )
    
    # Plot energy across time
    print("Generating energy vs time plots...")
    plot_energy_across_time(
        trial_statistics, 
        critical_energy, 
        output_dir,
        title_prefix="Full Reach"
    )
    
    # Look for transition points in the first stimulus
    print("Identifying transition points...")
    for i, energy_data in enumerate(analysis_results['energy_values']):
        if i < len(analysis_results['x_stim_data']):
            # Get mean energy and kinematics
            mean_energy = np.mean(energy_data, axis=0)
            mean_kinematics = np.mean(analysis_results['x_stim_data'][i], axis=0)
            
            # Save raw data
            raw_energy_df = pd.DataFrame(energy_data)
            raw_energy_df.to_csv(os.path.join(output_dir, f"raw_energy_stim_{i}.csv"), index=False)
            
            raw_kinematics_df = pd.DataFrame(analysis_results['x_stim_data'][i])
            raw_kinematics_df.to_csv(os.path.join(output_dir, f"raw_kinematics_stim_{i}.csv"), index=False)
            
            # Identify transition points
            transition_points = identify_transition_points(mean_energy, mean_kinematics)
            
            # Plot transition points
            if len(transition_points) > 0:
                plot_transition_points(
                    mean_energy, 
                    mean_kinematics,
                    transition_points,
                    output_dir,
                    stim_idx=i
                )
                
                print(f"Identified {len(transition_points)} transition points in stim_{i}")
            
            # Plot energy histogram
            plot_energy_histogram(
                mean_energy, 
                critical_energy, 
                output_dir, 
                stim_idx=i
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