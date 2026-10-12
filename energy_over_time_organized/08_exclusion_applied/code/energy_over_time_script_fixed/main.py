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
from coniii import *

# Import custom modules
from utils import load_and_preprocess_data, preprocessingSpikes, calculate_statistics_with_ci
from model import fit_ising_model, phase_transition_analysis, calculate_energy_for_spike_data, calculate_energy_by_neuron_count, calc_e
from analysis import create_energy_spline, analyze_neural_stimuli, analyze_single_stimulus, calculate_statistics_across_trials, identify_transition_points, identify_firing_rate_transition_points
from visualization import create_output_directory, plot_phase_transition, plot_energy_across_time, plot_transition_points, plot_energy_histogram, plot_model_quality, plot_model_quality_summary, plot_energy_distribution_by_k

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
    parser.add_argument('--eta', type=float, default=0.005,
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
    parser.add_argument('--firing_rate_window', type=int, default=10,
                        help='Window size for time-dependent firing rate calculation (default: 10)')
    parser.add_argument('--energy_by_k_samples', type=int, default=10000,
                        help='Number of random samples per k value for energy distribution analysis (default: 1000)')
    parser.add_argument('--skip_energy_by_k', action='store_true',
                        help='Skip energy distribution by k analysis')
    
    return parser.parse_args()

def _fit_and_save_model(bin_cat_p, N, args, stim_output_dir):
    """Fit the inverse Ising model to *bin_cat_p* and save h/J parameters."""
    from coniii.samplers import Metropolis

    print(f"  Fitting Ising model ({bin_cat_p.shape[0]} samples, {N} neurons)...")
    multipliers, solver = fit_ising_model(
        bin_cat_p,
        sample_size=args.sample_size,
        n_cpus=args.n_cpus,
        max_iter=args.max_iter,
        eta=args.eta,
    )

    # h parameters
    pd.DataFrame({'h_values': multipliers[:N]}).to_csv(
        os.path.join(stim_output_dir, "h_parameters.csv"), index=False)

    # J parameters (upper-triangle coupling matrix)
    J_params = multipliers[N:]
    J_dict, k = {}, 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            J_dict[f'J_{i+1}_{j+1}'] = J_params[k]
            k += 1
    pd.DataFrame([J_dict]).to_csv(os.path.join(stim_output_dir, "J_parameters.csv"), index=False)

    # Sample from the fitted model (needed for model-quality plots)
    print("  Sampling from fitted model...")
    m = Metropolis(N, solver.multipliers, calc_e)
    m.generate_sample_parallel_py(args.sample_size)

    return multipliers, solver, m


def _run_stim_pipeline(stim_idx, neural_stim_i, continous_stim_i,
                       multipliers, N, bin_cat_p, m, args, stim_output_dir):
    """
    Full analysis pipeline for a single stimulus using its own fitted model.

    Outputs written to *stim_output_dir*:
      preprocessing_info.csv, h_parameters.csv, J_parameters.csv
      model quality plots/CSVs, phase transition plots/CSVs
      per_reach_state.csv, energy/kinematic plots/CSVs
      transition point plots/CSVs, energy histograms
    """
    # ── Model quality ─────────────────────────────────────────────────────
    print(f"  [stim_{stim_idx}] Evaluating model quality...")
    plot_model_quality(bin_cat_p, m.sample, stim_output_dir)
    plot_model_quality_summary(bin_cat_p, m.sample, multipliers, N, stim_output_dir)

    # ── Energy by k ───────────────────────────────────────────────────────
    if not args.skip_energy_by_k:
        print(f"  [stim_{stim_idx}] Calculating energy distribution by k...")
        energy_by_k_results = calculate_energy_by_neuron_count(
            multipliers, N,
            num_samples_per_count=args.energy_by_k_samples,
            rng_seed=42,
            num_cores=args.n_cpus,
        )
        plot_energy_distribution_by_k(energy_by_k_results, stim_output_dir)
    else:
        print(f"  [stim_{stim_idx}] Skipping energy-by-k analysis.")

    # ── Phase transition ──────────────────────────────────────────────────
    if not args.skip_phase_analysis:
        print(f"  [stim_{stim_idx}] Analyzing phase transitions...")
        temp_range = np.arange(args.temp_min, args.temp_max, args.temp_step)
        pt_results = phase_transition_analysis(
            multipliers, N,
            temp_range=temp_range,
            samples=args.metropolis_samples,
            num_cores=args.n_cpus,
        )
        energy_spline  = plot_phase_transition(pt_results, stim_output_dir)
        critical_energy = pt_results['critical_energy']
        with open(os.path.join(stim_output_dir, "critical_values.txt"), "w") as f:
            f.write(f"Number of Neurons: {N}\n")
            f.write(f"Critical Temperature: {pt_results['critical_temp']}\n")
            f.write(f"Critical Energy: {critical_energy}\n")
    else:
        print(f"  [stim_{stim_idx}] Skipping phase transition analysis.")
        critical_energy = -5.0
        energy_spline   = None

    # ── Stimulus analysis with this stim's own model ──────────────────────
    print(f"  [stim_{stim_idx}] Running per-stimulus Ising analysis...")
    analysis_results = analyze_single_stimulus(
        neural_stim_i=neural_stim_i,
        continous_stim_i=continous_stim_i,
        multipliers=multipliers,
        critical_energy=critical_energy,
        stim_idx=stim_idx,
        fr_window=args.firing_rate_window,
        energy_temp_spline=energy_spline,
        output_dir=stim_output_dir,
    )

    # ── Trial statistics ──────────────────────────────────────────────────
    print(f"  [stim_{stim_idx}] Calculating trial statistics...")
    trial_statistics = calculate_statistics_across_trials(
        analysis_results,
        confidence=args.confidence,
    )

    # ── Energy-vs-time plots ──────────────────────────────────────────────
    print(f"  [stim_{stim_idx}] Generating energy vs time plots...")
    plot_energy_across_time(
        trial_statistics,
        critical_energy,
        stim_output_dir,
        title_prefix=f"Stim {stim_idx} — Full Reach",
        neural_data=[neural_stim_i],
        window_size=args.firing_rate_window,
        stim_idx_offset=stim_idx,
    )

    # ── Transition points ─────────────────────────────────────────────────
    print(f"  [stim_{stim_idx}] Identifying transition points...")
    energy_data     = analysis_results['energy_values'][0]
    mean_energy     = np.mean(energy_data, axis=0)
    mean_kinematics = np.mean(analysis_results['x_stim_data'][0], axis=0)

    pd.DataFrame(energy_data).to_csv(
        os.path.join(stim_output_dir, f"raw_energy_stim_{stim_idx}.csv"), index=False)
    pd.DataFrame(analysis_results['x_stim_data'][0]).to_csv(
        os.path.join(stim_output_dir, f"raw_kinematics_stim_{stim_idx}.csv"), index=False)

    energy_tp = identify_transition_points(
        mean_energy, mean_kinematics,
        output_dir=stim_output_dir, stim_idx=stim_idx,
    )

    fr_tp, mean_fr = [], None
    fr_list = analysis_results['firing_rate_values'][0]
    if fr_list:
        mean_fr = np.mean(np.array(fr_list), axis=0)
        fr_tp   = identify_firing_rate_transition_points(
            mean_fr, mean_kinematics,
            output_dir=stim_output_dir, stim_idx=stim_idx,
        )
        print(f"  [stim_{stim_idx}] {len(fr_tp)} firing-rate transition point(s) found.")

    plot_transition_points(
        mean_energy, mean_kinematics, energy_tp,
        stim_output_dir, stim_idx=stim_idx,
        firing_rate_transition_points=fr_tp if len(fr_tp) > 0 else None,
        firing_rate_data=mean_fr,
    )

    if len(energy_tp) > 0:
        print(f"  [stim_{stim_idx}] {len(energy_tp)} energy transition point(s) found.")

    plot_energy_histogram(mean_energy, critical_energy, stim_output_dir, stim_idx=stim_idx)


def run_analysis(args):
    """
    Full per-stimulus analysis pipeline.

    Each stimulus gets its OWN inverse Ising model fit and its own output
    subdirectory  <output_dir>/stim_<i>/.  Nothing is shared between stimuli.

    Output layout
    -------------
    <output_dir>/
      stim_0/
        preprocessing_info.csv
        h_parameters.csv, J_parameters.csv
        model_quality_summary*.csv/png, correlation_order_*.csv/png
        avg_spin_vs_temp.csv/png, heat_capacity.csv/png, energy_vs_temp.csv/png
        critical_values.txt/csv
        energy_distribution_by_k_*.csv/png  (unless --skip_energy_by_k)
        per_reach_state.csv
        neural_stimuli_summary.csv
        energy_stim_0.csv/png, x/y/z_kinematics_stim_0.csv, firing_rates_stim_0.csv
        raw_energy_stim_0.csv, raw_kinematics_stim_0.csv
        transition_points_stim_0.csv/png
        energy_histogram_stim_0.csv/png
      stim_1/   (same structure, independent model)
      stim_2/   (same structure, independent model)
    """
    if args.output_dir is None:
        args.output_dir = splitext(basename(args.matlab_file))[0]

    base_output_dir = create_output_directory(args.output_dir)

    # ── Load data once ────────────────────────────────────────────────────
    print(f"Loading data from {args.matlab_file}...")
    neural_stim, continous_stim = load_and_preprocess_data(
        args.matlab_file,
        truncate_idx_l=args.truncate_idx_l,
        truncate_idx=args.truncate_idx,
    )

    n_stims = len(neural_stim)
    print(f"Found {n_stims} stimulus condition(s). Fitting independent models for each.")

    # ── Per-stimulus loop ─────────────────────────────────────────────────
    for stim_idx in range(n_stims):
        print(f"\n{'='*60}")
        print(f"  STIMULUS {stim_idx}  ({stim_idx + 1}/{n_stims})")
        print(f"{'='*60}")

        stim_output_dir = create_output_directory(
            os.path.join(base_output_dir, f"stim_{stim_idx}")
        )

        # Preprocess this stimulus's spike data into {-1, +1}
        print(f"  [stim_{stim_idx}] Preprocessing spike data...")
        bin_cat   = np.vstack(neural_stim[stim_idx])
        bin_cat_p = preprocessingSpikes(bin_cat, args.bin_size)
        bin_cat_p = 2 * bin_cat_p - 1

        N = bin_cat_p.shape[1]
        print(f"  [stim_{stim_idx}] Neurons: {N}")

        pd.DataFrame([{
            'Number_of_Neurons':    N,
            'Bin_Size':             args.bin_size,
            'Truncate_Index_Lower': args.truncate_idx_l,
            'Truncate_Index_Upper': args.truncate_idx,
            'Stim_Index':           stim_idx,
            'Reach_Phase':          os.path.basename(base_output_dir),
        }]).to_csv(os.path.join(stim_output_dir, "preprocessing_info.csv"), index=False)

        # Fit inverse Ising, sample, save parameters
        multipliers, solver, m = _fit_and_save_model(
            bin_cat_p, N, args, stim_output_dir)

        # Run full pipeline for this stimulus with its own model
        _run_stim_pipeline(
            stim_idx=stim_idx,
            neural_stim_i=neural_stim[stim_idx],
            continous_stim_i=continous_stim[stim_idx],
            multipliers=multipliers,
            N=N,
            bin_cat_p=bin_cat_p,
            m=m,
            args=args,
            stim_output_dir=stim_output_dir,
        )

        print(f"  [stim_{stim_idx}] Done. Results in: {stim_output_dir}")

    print(f"\nAll {n_stims} stimulus pipeline(s) complete.")
    print(f"Results saved under: {base_output_dir}")

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