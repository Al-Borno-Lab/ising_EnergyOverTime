#!/usr/bin/env python
# coding: utf-8

"""
Visualization functions for the Ising model analysis.
This module contains functions for creating and saving plots of model results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from utils import calculate_time_dependent_firing_rate

def create_output_directory(output_dir):
    """
    Create output directory for saving plots.
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory
        
    Returns:
    --------
    str
        Path to created directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

def plot_phase_transition(results, output_dir):
    """
    Create plots for phase transition analysis.
    
    Parameters:
    -----------
    results : dict
        Results from phase transition analysis
    output_dir : str
        Directory to save plots
        
    Returns:
    --------
    CubicSpline
        Spline function that maps temperature to energy
    """
    # Extract data from results
    temp_range = results['temp_range']
    pos_avg_pos = results['pos_avg_pos']
    neg_avg_pos = results['neg_avg_pos']
    pos_avg_energy = results['pos_avg_energy']
    neg_avg_energy = results['neg_avg_energy']
    avg_heat_capacity_pos = results['avg_heat_capacity_pos']
    avg_heat_capacity_neg = results['avg_heat_capacity_neg']
    c_temp = results['critical_temp']
    critical_energy = results['critical_energy']
    
    # Plot 1: Average spin vs temperature
    plt.figure(figsize=(10, 6))
    plt.title("Average Spin per Temperature")
    plt.plot(temp_range, pos_avg_pos, c='blue', label='Positive initialization')
    plt.plot(temp_range, neg_avg_pos, c='orange', label='Negative initialization')
    
    # Resting point - average of last 10 points
    conv_bar = np.mean(neg_avg_pos[-10:])
    plt.axhline(y=conv_bar, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel("Temperature")
    plt.ylabel("Average Spin / N²")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "avg_spin_vs_temp.png"))
    plt.close()
    
    # Save data to CSV
    spin_df = pd.DataFrame({
        'Temperature': temp_range,
        'Positive_Initialization_Spin': pos_avg_pos,
        'Negative_Initialization_Spin': neg_avg_pos
    })
    spin_df.to_csv(os.path.join(output_dir, "avg_spin_vs_temp.csv"), index=False)
    
    # Plot 2: Heat capacity
    plt.figure(figsize=(10, 6))
    plt.title("Heat Capacity vs Temperature")
    plt.plot(temp_range, avg_heat_capacity_pos, c='blue', label='Positive initialization')
    plt.plot(temp_range, avg_heat_capacity_neg, c='orange', label='Negative initialization')
    
    # Mark critical temperature
    plt.axvline(x=c_temp, color='r', linestyle='--')
    plt.text(c_temp + 0.05, max(avg_heat_capacity_pos) * 0.8, 
             f'Critical T = {c_temp:.2f}', rotation=90, color='r')
    
    plt.xlabel("Temperature")
    plt.ylabel("Heat Capacity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "heat_capacity.png"))
    plt.close()
    
    # Save data to CSV
    heat_df = pd.DataFrame({
        'Temperature': temp_range,
        'Positive_Initialization_Heat_Capacity': avg_heat_capacity_pos,
        'Negative_Initialization_Heat_Capacity': avg_heat_capacity_neg
    })
    heat_df.to_csv(os.path.join(output_dir, "heat_capacity.csv"), index=False)
    
    # Plot 3: Energy vs temperature
    plt.figure(figsize=(10, 6))
    plt.title("Energy per Temperature")
    plt.plot(temp_range, pos_avg_energy, c='blue', label='Positive initialization')
    plt.plot(temp_range, neg_avg_energy, c='orange', label='Negative initialization')
    
    # Mark critical temperature and energy
    plt.axvline(x=c_temp, color='r', linestyle='--')
    plt.axhline(y=critical_energy, color='g', linestyle='--')
    plt.text(c_temp + 0.05, min(pos_avg_energy + neg_avg_energy), 
             f'Critical T = {c_temp:.2f}', rotation=90, color='r')
    plt.text(temp_range[0] + 0.05, critical_energy + 0.2, 
             f'Critical E = {critical_energy:.2f}', color='g')
    
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "energy_vs_temp.png"))
    plt.close()
    
    # Save data to CSV
    energy_df = pd.DataFrame({
        'Temperature': temp_range,
        'Positive_Initialization_Energy': pos_avg_energy,
        'Negative_Initialization_Energy': neg_avg_energy
    })
    energy_df.to_csv(os.path.join(output_dir, "energy_vs_temp.csv"), index=False)
    
    # Save critical values to their own CSV
    critical_df = pd.DataFrame({
        'Phase': [os.path.basename(output_dir)],
        'Critical_Temperature': [c_temp],
        'Critical_Energy': [critical_energy]
    })
    
    # Determine if it's a reach phase directory
    base_dir = os.path.dirname(output_dir)
    csv_path = os.path.join(base_dir, "critical_values.csv")
    
    # If the CSV exists, append to it, otherwise create it
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, critical_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        critical_df.to_csv(csv_path, index=False)
    
    # Also save to the phase directory
    critical_df.to_csv(os.path.join(output_dir, "critical_values.csv"), index=False)
    
    print(f"Phase transition plots and CSVs saved to {output_dir}")
    
    return CubicSpline(temp_range, pos_avg_energy)

def plot_energy_across_time(stats, critical_energy, output_dir, title_prefix="", show_mid_point=True, neural_data=None, window_size=10):
    """
    Plot energy and kinematic data across time.
    
    Parameters:
    -----------
    stats : dict
        Statistics calculated across trials
    critical_energy : float
        Critical energy from phase transition analysis
    output_dir : str
        Directory to save plots
    title_prefix : str, optional
        Prefix for plot titles (default="")
    show_mid_point : bool, optional
        Whether to mark the middle point (default=True)
    neural_data : list, optional
        List of neural data arrays for calculating firing rates (default=None)
    window_size : int, optional
        Size of the sliding window for firing rate calculation (default=10)
        
    Returns:
    --------
    None
    """
    kinematics = stats['kinematics']
    energy = stats['energy']
    
    for i, (kin, eng) in enumerate(zip(kinematics, energy)):
        # Create figure with three subplots if neural data is provided
        n_subplots = 3 if neural_data is not None else 2
        plt.figure(figsize=(12, 4*n_subplots))
        
        # Top subplot for kinematics
        plt.subplot(n_subplots, 1, 1)
        plt.title(f"{title_prefix} Kinematic Component Over Time, Stim_{i}")
        
        plt.plot(kin['mean'], '-r', label='mean')
        plt.plot(kin['upper'], '-b', label='upper', alpha=0.15)
        plt.plot(kin['lower'], '-b', label='lower', alpha=0.15)
        
        # Plot other stimulus means for comparison if available
        for j, other_kin in enumerate(kinematics):
            if j != i:
                plt.plot(other_kin['mean'], '-', label=f"mean_stim_{j}", alpha=0.7)
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(kin['mean']))), kin['upper'], kin['lower'], 
                         color="k", alpha=0.15)
        
        # Mark midpoint if requested
        if show_mid_point and len(kin['mean']) > 100:
            mid_point = len(kin['mean']) // 2
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Middle subplot for energy
        plt.subplot(n_subplots, 1, 2)
        plt.title(f"{title_prefix} Energy of Neural Activity Over Time, Stim_{i}")
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(eng['mean']))), eng['upper'], eng['lower'], 
                         color="k", alpha=0.15)
        
        # Plot other stimulus means for comparison if available
        for j, other_eng in enumerate(energy):
            if j != i:
                plt.plot(other_eng['mean'], '-', label=f"mean_stim_{j}", alpha=0.7)
        
        # Mark critical energy and mean energy
        plt.axhline(y=critical_energy, color='r', linestyle='--',
                   label=f"Critical Energy = {critical_energy:.2f}")
        plt.axhline(y=np.mean(eng['mean']), color='b', linestyle='-.',
                   label=f"Mean Energy = {np.mean(eng['mean']):.2f}")
        
        # Mark midpoint if requested
        if show_mid_point and len(eng['mean']) > 100:
            mid_point = len(eng['mean']) // 2
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7)
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Bottom subplot for firing rates if neural data is provided
        if neural_data is not None:
            plt.subplot(n_subplots, 1, 3)
            plt.title(f"{title_prefix} Time-Dependent Firing Rate (window={window_size}), Stim_{i}")
            
            # Calculate firing rates for each trial
            firing_rates = []
            for trial in neural_data[i]:
                # Calculate time-dependent firing rate
                rate = calculate_time_dependent_firing_rate(trial, window_size)
                # Ensure rate is 1-dimensional
                rate = np.squeeze(rate)
                firing_rates.append(rate)
            
            # Calculate mean and confidence intervals
            firing_rates = np.array(firing_rates)
            mean_firing = np.mean(firing_rates, axis=0)
            std_firing = np.std(firing_rates, axis=0)
            ci_firing = 1.96 * std_firing / np.sqrt(firing_rates.shape[0])
            
            # Ensure all arrays are 1-dimensional
            mean_firing = np.squeeze(mean_firing)
            ci_firing = np.squeeze(ci_firing)
            
            # Plot mean firing rate and confidence intervals
            plt.plot(mean_firing, '-r', label='mean')
            plt.fill_between(range(len(mean_firing)), 
                           mean_firing - ci_firing,
                           mean_firing + ci_firing,
                           color='k', alpha=0.15)
            
            # Mark midpoint if requested
            if show_mid_point and len(mean_firing) > 100:
                mid_point = len(mean_firing) // 2
                plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7)
                plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
            
            plt.xlabel("Time")
            plt.ylabel("Firing Rate")
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save firing rate data to CSV
            firing_df = pd.DataFrame({
                'Time': range(len(mean_firing)),
                'Mean_Firing_Rate': mean_firing,
                'Upper_CI': mean_firing + ci_firing,
                'Lower_CI': mean_firing - ci_firing,
                'Window_Size': [window_size] * len(mean_firing)
            })
            firing_df.to_csv(os.path.join(output_dir, f"firing_rates_stim_{i}.csv"), index=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"energy_kinematics_stim_{i}.png"))
        plt.close()
        
        # Save data to CSV
        time_points = list(range(len(kin['mean'])))
        
        # Kinematics CSV
        kin_df = pd.DataFrame({
            'Time': time_points,
            'Mean_Position': kin['mean'],
            'Upper_CI': kin['upper'],
            'Lower_CI': kin['lower']
        })
        
        # Add other stim means for comparison
        for j, other_kin in enumerate(kinematics):
            if j != i and len(other_kin['mean']) == len(kin['mean']):
                kin_df[f'Mean_Position_Stim_{j}'] = other_kin['mean']
        
        kin_df.to_csv(os.path.join(output_dir, f"kinematics_stim_{i}.csv"), index=False)
        
        # Energy CSV
        eng_df = pd.DataFrame({
            'Time': time_points,
            'Mean_Energy': eng['mean'],
            'Upper_CI': eng['upper'],
            'Lower_CI': eng['lower'],
            'Critical_Energy': [critical_energy] * len(eng['mean']),
            'Mean_Energy_Overall': [np.mean(eng['mean'])] * len(eng['mean'])
        })
        
        # Add other stim means for comparison
        for j, other_eng in enumerate(energy):
            if j != i and len(other_eng['mean']) == len(eng['mean']):
                eng_df[f'Mean_Energy_Stim_{j}'] = other_eng['mean']
        
        eng_df.to_csv(os.path.join(output_dir, f"energy_stim_{i}.csv"), index=False)
    
    print(f"Energy and kinematic plots and CSVs saved to {output_dir}")

def plot_transition_points(energy_data, kinematic_data, transition_points, output_dir, stim_idx=0):
    """
    Plot transition points identified in the neural activity.
    
    Parameters:
    -----------
    energy_data : ndarray
        Energy values over time
    kinematic_data : ndarray
        Kinematic values over time
    transition_points : list
        Indices of identified transition points
    output_dir : str
        Directory to save plots
    stim_idx : int, optional
        Stimulus index (default=0)
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=(12, 10))
    
    # Top subplot for kinematics
    plt.subplot(2, 1, 1)
    plt.title(f"Kinematic Data with Transition Points, Stim_{stim_idx}")
    plt.plot(kinematic_data, '-b')
    
    # Mark transition points
    for point in transition_points:
        plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.grid(alpha=0.3)
    
    # Bottom subplot for energy
    plt.subplot(2, 1, 2)
    plt.title(f"Energy Data with Transition Points, Stim_{stim_idx}")
    plt.plot(energy_data, '-g')
    
    # Mark transition points
    for point in transition_points:
        plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
        if point < len(energy_data):
            plt.plot(point, energy_data[point], 'ro', markersize=8)
    
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"transition_points_stim_{stim_idx}.png"))
    plt.close()
    
    # Save data to CSV
    time_points = list(range(len(kinematic_data)))
    
    # Create a dataframe with time, kinematic data, and energy data
    transitions_df = pd.DataFrame({
        'Time': time_points,
        'Kinematic_Data': kinematic_data,
        'Energy_Data': energy_data,
    })
    
    # Add a column indicating if the point is a transition point
    transitions_df['Is_Transition_Point'] = [1 if i in transition_points else 0 for i in time_points]
    
    # Save to CSV
    transitions_df.to_csv(os.path.join(output_dir, f"transition_points_stim_{stim_idx}.csv"), index=False)
    
    print(f"Transition points plot and CSV saved to {output_dir}")

def plot_energy_histogram(energy_values, critical_energy, output_dir, stim_idx=0, bins=50):
    """
    Plot histogram of energy values for neural activity.
    
    Parameters:
    -----------
    energy_values : ndarray
        Energy values to plot
    critical_energy : float
        Critical energy from phase transition analysis
    output_dir : str
        Directory to save plots
    stim_idx : int, optional
        Stimulus index (default=0)
    bins : int, optional
        Number of bins for histogram (default=50)
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Distribution of Neural Energy Values, Stim_{stim_idx}")
    
    # Plot histogram
    hist, bin_edges = np.histogram(energy_values, bins=bins, density=True)
    plt.hist(energy_values, bins=bins, alpha=0.7, color='steelblue', density=True)
    
    # Mark critical energy
    plt.axvline(x=critical_energy, color='r', linestyle='--',
               label=f"Critical Energy = {critical_energy:.2f}")
    
    # Mark mean energy
    mean_energy = np.mean(energy_values)
    plt.axvline(x=mean_energy, color='g', linestyle='-.',
               label=f"Mean Energy = {mean_energy:.2f}")
    
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f"energy_histogram_stim_{stim_idx}.png"))
    plt.close()
    
    # Save histogram data to CSV
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    hist_df = pd.DataFrame({
        'Bin_Center': bin_centers,
        'Density': hist,
        'Bin_Left_Edge': bin_edges[:-1],
        'Bin_Right_Edge': bin_edges[1:],
    })
    
    # Add critical and mean energy values
    hist_df['Critical_Energy'] = critical_energy
    hist_df['Mean_Energy'] = mean_energy
    
    # Save to CSV
    hist_df.to_csv(os.path.join(output_dir, f"energy_histogram_stim_{stim_idx}.csv"), index=False)
    
    print(f"Energy histogram and CSV saved to {output_dir}")

def plot_model_quality(original_data, model_samples, output_dir, max_corr_order=6):
    """
    Plot comparison of correlations between original data and model samples.
    
    Parameters:
    -----------
    original_data : ndarray
        Original spike data
    model_samples : ndarray
        Samples generated from the model
    output_dir : str
        Directory to save plots
    max_corr_order : int, optional
        Maximum correlation order to evaluate (default=6)
        
    Returns:
    --------
    None
    """
    from coniii.utils import k_corr
    
    for i in range(2, max_corr_order + 1):
        plt.figure(figsize=(8, 6))
        plt.title(f"{i}-point Correlation Comparison")
        
        # Calculate correlations
        original_corr = k_corr(original_data, i)
        model_corr = k_corr(model_samples, i)
        
        # Plot comparison
        plt.scatter(original_corr, model_corr, alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', label="Perfect Match")
        
        plt.xlabel("Original Data Correlation")
        plt.ylabel("Model Correlation")
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(original_corr, model_corr)[0, 1]
        plt.text(0.05, 0.95, f"Correlation: {corr_coef:.4f}", 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, f"correlation_order_{i}.png"))
        plt.close()
        
        # Save correlation data to CSV
        corr_df = pd.DataFrame({
            'Original_Correlation': original_corr,
            'Model_Correlation': model_corr
        })
        corr_df['Correlation_Coefficient'] = corr_coef
        corr_df.to_csv(os.path.join(output_dir, f"correlation_order_{i}.csv"), index=False)
    
    print(f"Model quality plots and CSVs saved to {output_dir}")