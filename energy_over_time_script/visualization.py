#!/usr/bin/env python
# coding: utf-8

"""
Visualization functions for the Ising model analysis.
This module contains functions for creating and saving plots of model results.
"""

import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    Plot energy and kinematic data (x, y, z coordinates) across time.
    
    Parameters:
    -----------
    stats : dict
        Statistics calculated across trials containing x_kinematics, y_kinematics, z_kinematics, energy
    critical_energy : float
        Critical energy from phase transition analysis
    output_dir : str
        Directory to save plots
    title_prefix : str, optional
        Prefix for plot titles (default="")
    show_mid_point : bool, optional
        Whether to mark the middle point (default=True)
    neural_data : list, optional
        List of neural data arrays where each stimulus contains trials of binary spike vectors (default=None)
    window_size : int, optional
        Size of the sliding window for firing rate calculation (default=10)
        
    Returns:
    --------
    None
    """
    x_kinematics = stats['x_kinematics']
    y_kinematics = stats['y_kinematics']
    z_kinematics = stats['z_kinematics']
    energy = stats['energy']
    j_values = stats.get('j_values', None)
    h_values = stats.get('h_values', None)

    
    for i, (x_kin, y_kin, z_kin, eng) in enumerate(zip(x_kinematics, y_kinematics, z_kinematics, energy)):
        # Create figure with stacked subplots: X, Y, Z, Energy, and optionally Firing Rate
        n_subplots = 6 if neural_data is not None else 4
        plt.figure(figsize=(12, 3*n_subplots))
        
        # X-coordinate subplot
        plt.subplot(n_subplots, 1, 1)
        plt.title(f"{title_prefix} X-Coordinate Over Time, Stim_{i}")
        
        plt.plot(x_kin['mean'], '-r', linewidth=2, label='mean X')
        plt.plot(x_kin['upper'], '-b', label='upper CI', alpha=0.5)
        plt.plot(x_kin['lower'], '-b', label='lower CI', alpha=0.5)
        
        # Plot other stimulus means for comparison if available
        for j, other_x_kin in enumerate(x_kinematics):
            if j != i:
                plt.plot(other_x_kin['mean'], '-', label=f"mean X stim_{j}", alpha=0.7)
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(x_kin['mean']))), x_kin['upper'], x_kin['lower'], 
                         color="blue", alpha=0.15)
        
        # Mark midpoint if requested
        if show_mid_point and len(x_kin['mean']) > 100:
            mid_point = 400
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.ylabel("X Position")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Y-coordinate subplot
        plt.subplot(n_subplots, 1, 2)
        plt.title(f"{title_prefix} Y-Coordinate Over Time, Stim_{i}")
        
        plt.plot(y_kin['mean'], '-g', linewidth=2, label='mean Y')
        plt.plot(y_kin['upper'], '-b', label='upper CI', alpha=0.5)
        plt.plot(y_kin['lower'], '-b', label='lower CI', alpha=0.5)
        
        # Plot other stimulus means for comparison if available
        for j, other_y_kin in enumerate(y_kinematics):
            if j != i:
                plt.plot(other_y_kin['mean'], '-', label=f"mean Y stim_{j}", alpha=0.7)
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(y_kin['mean']))), y_kin['upper'], y_kin['lower'], 
                         color="green", alpha=0.15)
        
        # Mark midpoint if requested
        if show_mid_point and len(y_kin['mean']) > 100:
            mid_point = 400
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Z-coordinate subplot
        plt.subplot(n_subplots, 1, 3)
        plt.title(f"{title_prefix} Z-Coordinate Over Time, Stim_{i}")
        
        plt.plot(z_kin['mean'], '-r', linewidth=2, label='mean Z', color='red')
        plt.plot(z_kin['upper'], '-b', label='upper CI', alpha=0.5)
        plt.plot(z_kin['lower'], '-b', label='lower CI', alpha=0.5)
        
        # Plot other stimulus means for comparison if available
        for j, other_z_kin in enumerate(z_kinematics):
            if j != i:
                plt.plot(other_z_kin['mean'], '-', label=f"mean Z stim_{j}", alpha=0.7)
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(z_kin['mean']))), z_kin['upper'], z_kin['lower'], 
                         color="red", alpha=0.15)
        
        # Mark midpoint if requested
        if show_mid_point and len(z_kin['mean']) > 100:
            mid_point = 400
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.ylabel("Z Position")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Energy subplot --
        plt.subplot(n_subplots, 1, 4)
        plt.title(f"{title_prefix} Energy of Neural Activity Over Time, Stim_{i}")
        
        # Fill between confidence intervals
        plt.fill_between(list(range(len(eng['mean']))), eng['upper'], eng['lower'], 
                         color="purple", alpha=0.15)
        
        plt.plot(eng['mean'], '-', linewidth=2, label='mean Energy', color='purple')
        
        # Plot other stimulus means for comparison if available
        for j, other_eng in enumerate(energy):
            if j != i:
                plt.plot(other_eng['mean'], '-', label=f"mean Energy stim_{j}", alpha=0.7)
        
        # Mark critical energy and mean energy
        plt.axhline(y=critical_energy, color='r', linestyle='--',
                   label=f"Critical Energy = {critical_energy:.2f}")
        plt.axhline(y=np.mean(eng['mean']), color='b', linestyle='-.',
                   label=f"Mean Energy = {np.mean(eng['mean']):.2f}")
        
        # Mark midpoint if requested
        if show_mid_point and len(eng['mean']) > 100:
            mid_point = 400
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)
        
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # h & j plots subplot
        plt.subplot(n_subplots, 1, 5)

        # plot 
        plt.title(f"{title_prefix} Local Fields and Interactions, Stim_{i}")
        
        if show_mid_point and len(j_values[i]['mean']) > 100:
            mid_point = 400
            plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
            plt.axvline(x=mid_point - 25, color='g', linestyle=':', alpha=0.5)

        
        plt.plot(j_values[i]['mean'], '-', label=f"[j] - mean Energy stim_{i}")
        plt.plot(h_values[i]['mean'], '-', label=f"[h] - mean Energy stim_{i}")

        plt.ylabel("Energy")
        plt.legend()
        plt.grid(alpha=0.3)

        # Firing rate subplot if neural data is provided
        if neural_data is not None:
            plt.subplot(n_subplots, 1, 6)
            plt.title(f"{title_prefix} Time-Dependent Firing Rate (window={window_size}), Stim_{i}")
            
            # Calculate firing rates for each trial in the current stimulus
            firing_rates = []
            for trial in neural_data[i]:
                # Calculate time-dependent firing rate for this trial
                # trial shape is (time_bins, num_neurons)
                # We want the mean firing rate across all neurons
                rate = calculate_time_dependent_firing_rate(trial, window_size)
                # Ensure rate is 1-dimensional
                rate = np.squeeze(rate)
                firing_rates.append(rate)
            
            # Convert to numpy array for easier computation
            firing_rates = np.array(firing_rates)
            
            # Calculate mean and confidence intervals across trials
            mean_firing = np.mean(firing_rates, axis=0)
            std_firing = np.std(firing_rates, axis=0)
            ci_firing = 1.96 * std_firing / np.sqrt(firing_rates.shape[0])
            
            # Ensure all arrays are 1-dimensional
            mean_firing = np.squeeze(mean_firing)
            ci_firing = np.squeeze(ci_firing)
            
            # Plot mean firing rate and confidence intervals
            plt.plot(mean_firing, '-', linewidth=2, label='mean Firing Rate', color='orange')
            plt.fill_between(range(len(mean_firing)), 
                           mean_firing - ci_firing,
                           mean_firing + ci_firing,
                           color='orange', alpha=0.15)
            
            # Mark midpoint if requested
            if show_mid_point and len(mean_firing) > 100:
                mid_point = 400
                plt.axvline(x=mid_point, color='g', linestyle='--', alpha=0.7, label='midpoint')
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
        else:
            # Add x-label to the last subplot if no firing rate subplot
            plt.xlabel("Time")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"energy_kinematics_stim_{i}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV files
        time_points = list(range(len(x_kin['mean'])))
        
        # X-coordinate CSV
        x_df = pd.DataFrame({
            'Time': time_points,
            'Mean_X_Position': x_kin['mean'],
            'Upper_CI': x_kin['upper'],
            'Lower_CI': x_kin['lower']
        })
        
        # Add other stim means for comparison
        for j, other_x_kin in enumerate(x_kinematics):
            if j != i and len(other_x_kin['mean']) == len(x_kin['mean']):
                x_df[f'Mean_X_Position_Stim_{j}'] = other_x_kin['mean']
        
        x_df.to_csv(os.path.join(output_dir, f"x_kinematics_stim_{i}.csv"), index=False)
        
        # Y-coordinate CSV
        y_df = pd.DataFrame({
            'Time': time_points,
            'Mean_Y_Position': y_kin['mean'],
            'Upper_CI': y_kin['upper'],
            'Lower_CI': y_kin['lower']
        })
        
        # Add other stim means for comparison
        for j, other_y_kin in enumerate(y_kinematics):
            if j != i and len(other_y_kin['mean']) == len(y_kin['mean']):
                y_df[f'Mean_Y_Position_Stim_{j}'] = other_y_kin['mean']
        
        y_df.to_csv(os.path.join(output_dir, f"y_kinematics_stim_{i}.csv"), index=False)
        
        # Z-coordinate CSV
        z_df = pd.DataFrame({
            'Time': time_points,
            'Mean_Z_Position': z_kin['mean'],
            'Upper_CI': z_kin['upper'],
            'Lower_CI': z_kin['lower']
        })
        
        # Add other stim means for comparison
        for j, other_z_kin in enumerate(z_kinematics):
            if j != i and len(other_z_kin['mean']) == len(z_kin['mean']):
                z_df[f'Mean_Z_Position_Stim_{j}'] = other_z_kin['mean']
        
        z_df.to_csv(os.path.join(output_dir, f"z_kinematics_stim_{i}.csv"), index=False)
        
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
    
    print(f"Energy and kinematic (X, Y, Z) plots and CSVs saved to {output_dir}")

def plot_transition_points(energy_data, kinematic_data, energy_transition_points, output_dir, stim_idx=0,
                          firing_rate_transition_points=None, firing_rate_data=None):
    """
    Plot transition points on position, velocity, acceleration, energy, and firing rate.
    
    Compares energy-based and firing-rate-based transition points with different colors
    to visualize which kinematic transitions are detected by each signal.
    
    Parameters:
    -----------
    energy_data : ndarray
        Energy values over time
    kinematic_data : ndarray
        Position (kinematic) values over time
    energy_transition_points : list
        Indices of transition points identified from energy
    output_dir : str
        Directory to save plots
    stim_idx : int, optional
        Stimulus index (default=0)
    firing_rate_transition_points : list, optional
        Indices of transition points identified from firing rate (default=None).
        When provided, both types are plotted with different colors for comparison.
    firing_rate_data : ndarray, optional
        Firing rate values over time for plotting (default=None).
        When provided, firing rate is plotted below the kinematic graphs.
        
    Returns:
    --------
    None
    """
    kinematic_data = np.array(kinematic_data)
    energy_data = np.array(energy_data)
    
    # Compute velocity and acceleration from position
    velocity = np.gradient(kinematic_data)
    acceleration = np.gradient(velocity)
    
    # Align lengths (energy may differ slightly)
    n_points = len(kinematic_data)
    energy_data = energy_data[:n_points] if len(energy_data) > n_points else np.pad(
        energy_data, (0, max(0, n_points - len(energy_data))), mode='edge'
    )[:n_points]
    
    if firing_rate_transition_points is None:
        firing_rate_transition_points = []
    
    # Align firing rate data if provided
    firing_rate_data = np.array(firing_rate_data) if firing_rate_data is not None else None
    if firing_rate_data is not None:
        firing_rate_data = firing_rate_data[:n_points] if len(firing_rate_data) > n_points else np.pad(
            firing_rate_data, (0, max(0, n_points - len(firing_rate_data))), mode='edge'
        )[:n_points]
    
    # Find overlap: points detected by both energy and firing rate
    energy_set = set(energy_transition_points)
    fr_set = set(firing_rate_transition_points)
    both_set = energy_set & fr_set
    
    # 5 subplots: position, velocity, acceleration, energy, firing rate (if available)
    n_subplots = 5 if firing_rate_data is not None else 4
    plt.figure(figsize=(14, 2.5 * n_subplots))
    
    time_axis = np.arange(n_points)
    
    def _add_transition_lines():
        for pt in energy_transition_points:
            if pt < n_points:
                plt.axvline(x=pt, color='red', linestyle='--', alpha=0.7, linewidth=1)
        for pt in firing_rate_transition_points:
            if pt < n_points:
                plt.axvline(x=pt, color='blue', linestyle=':', alpha=0.7, linewidth=1)
        for pt in both_set:
            if pt < n_points:
                plt.axvline(x=pt, color='purple', linestyle='-', alpha=0.9, linewidth=1.5)
    
    # Subplot 1: Position
    plt.subplot(n_subplots, 1, 1)
    plt.title(f"Position, Velocity, Acceleration, Energy, and Firing Rate with Transition Points, Stim_{stim_idx}")
    plt.plot(time_axis, kinematic_data, '-b', linewidth=1.5, label='Position')
    _add_transition_lines()
    plt.ylabel("Position")
    plt.legend(handles=[
        plt.Line2D([0], [0], color='red', linestyle='--', label='Energy transition'),
        plt.Line2D([0], [0], color='blue', linestyle=':', label='Firing rate transition'),
        plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='Both (shared)')
    ], loc='upper right')
    plt.grid(alpha=0.3)
    
    # Subplot 2: Velocity
    plt.subplot(n_subplots, 1, 2)
    plt.plot(time_axis, velocity, '-g', linewidth=1.5, label='Velocity')
    _add_transition_lines()
    plt.ylabel("Velocity")
    plt.grid(alpha=0.3)
    
    # Subplot 3: Acceleration
    plt.subplot(n_subplots, 1, 3)
    plt.plot(time_axis, acceleration, '-m', linewidth=1.5, label='Acceleration')
    _add_transition_lines()
    plt.ylabel("Acceleration")
    plt.grid(alpha=0.3)
    
    # Subplot 4: Energy
    plt.subplot(n_subplots, 1, 4)
    plt.plot(time_axis, energy_data, '-', color='darkgreen', linewidth=1.5, label='Energy')
    _add_transition_lines()
    plt.ylabel("Energy")
    if firing_rate_data is None:
        plt.xlabel("Time")
    plt.grid(alpha=0.3)
    
    # Subplot 5: Firing Rate (if available)
    if firing_rate_data is not None:
        plt.subplot(n_subplots, 1, 5)
        plt.plot(time_axis, firing_rate_data, '-', color='orange', linewidth=1.5, label='Firing Rate')
        _add_transition_lines()
        plt.xlabel("Time")
        plt.ylabel("Firing Rate")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"transition_points_stim_{stim_idx}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV with position, velocity, acceleration, energy, firing rate, and both transition types
    time_points = list(range(n_points))
    csv_data = {
        'Time': time_points,
        'Position': kinematic_data,
        'Velocity': velocity,
        'Acceleration': acceleration,
        'Energy_Data': energy_data[:n_points],
        'Energy_Transition': [1 if i in energy_transition_points else 0 for i in range(n_points)],
        'Firing_Rate_Transition': [1 if i in firing_rate_transition_points else 0 for i in range(n_points)],
        'Both_Transition': [1 if i in both_set else 0 for i in range(n_points)]
    }
    if firing_rate_data is not None:
        csv_data['Firing_Rate_Data'] = firing_rate_data[:n_points]
    transitions_df = pd.DataFrame(csv_data)
    
    transitions_df.to_csv(os.path.join(output_dir, f"transition_points_stim_{stim_idx}.csv"), index=False)
    
    print(f"Transition points plot and CSV saved to {output_dir} (energy: {len(energy_transition_points)}, "
          f"firing rate: {len(firing_rate_transition_points)}, shared: {len(both_set)})")

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

def plot_model_quality_summary(original_data, model_samples, multipliers, N, output_dir,
                               filename="model_quality_summary.png",
                               max_k_plot=None):
    """
    Single figure: pairwise correlations (k=2), J distribution, triplet correlations (k=3),
    and P(K) simultaneous spikes (data vs Ising vs independent Poisson-binomial).

    **P(K) procedure (panel d)**

    1. **Binary spikes** per bin: ``σ ∈ {0,1}^N`` (from ``{-1,+1}`` via ``(s+1)/2`` if needed).
    2. **Data:** ``K(t) = Σᵢ σᵢ``; ``P_data(K) = (# bins with K spikes) / (total bins)``.
    3. **Independent model:** marginal rates from data ``p_i = ⟨σᵢ⟩``, then Poisson-binomial
       ``P_ind(K)`` = coefficient of ``z^K`` in ``∏ᵢ [(1-p_i) + p_i z]``, computed by
       convolving Bernoulli masses (same as sum of independent Bernoullis).
    4. **Ising:** same histogram of ``K`` from Monte Carlo samples ``model_samples``.

    Parameters
    ----------
    original_data : ndarray
        Observed data (n_bins, N): spins in ``{-1, 1}`` or binary ``{0, 1}``.
    model_samples : ndarray
        Metropolis samples from the fitted Ising model (same convention as ``original_data``).
    multipliers : ndarray
        Full parameter vector [h_1..h_N, J_ij...].
    N : int
        Number of neurons.
    output_dir : str
        Output directory.
    filename : str
        Output PNG filename.
    max_k_plot : int, optional
        If set, x-axis is limited to ``[0, max_k_plot]`` (e.g. 20 for large N). Default ``None`` uses ``0 … N``.

    Writes CSVs to ``output_dir``:
        ``model_quality_summary_h_fields.csv``, ``model_quality_summary_J_couplings.csv``, …;
        ``model_quality_summary_P_K.csv`` — K, counts, P_data, P_ising, P_independent;
        ``model_quality_summary_P_K_metadata.csv`` — N, sample sizes;
        ``model_quality_summary_independent_marginals.csv`` — per-neuron empirical ``p_i = ⟨σᵢ⟩``;
        plus pairwise and triplet correlation CSVs.

    Returns
    -------
    None
    """
    from coniii.utils import k_corr

    original_data = np.asarray(original_data)
    model_samples = np.asarray(model_samples)
    h_params = np.asarray(multipliers[:N], dtype=float)
    J_params = np.asarray(multipliers[N:])

    corr2_orig = k_corr(original_data, 2)
    corr2_model = k_corr(model_samples, 2)
    corr3_orig = k_corr(original_data, 3)
    corr3_model = k_corr(model_samples, 3)

    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.15],
                           width_ratios=[1, 1], hspace=0.35, wspace=0.28)

    # --- (a) Pairwise k-corr: full width ---
    ax_pair = fig.add_subplot(gs[0, :])
    ax_pair.scatter(corr2_orig, corr2_model, s=12, alpha=0.5, c="k", edgecolors="none")
    lo = float(min(corr2_orig.min(), corr2_model.min(), 0))
    hi = float(max(corr2_orig.max(), corr2_model.max(), 1))
    ax_pair.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax_pair.set_xlabel(r"measured $C_{ij}$")
    ax_pair.set_ylabel(r"reconstructed $C_{ij}$")
    ax_pair.set_title("Pairwise correlation ($k=2$)")
    ax_pair.grid(alpha=0.3)
    ax_pair.text(0.02, 0.98, "(a)", transform=ax_pair.transAxes, fontsize=12, fontweight="bold",
                 va="top", ha="left")

    # Inset: zoom near origin (small correlations)
    try:
        ax_in = ax_pair.inset_axes([0.55, 0.08, 0.42, 0.38])
        ax_in.scatter(corr2_orig, corr2_model, s=8, alpha=0.45, c="gray")
        ax_in.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        lim = max(0.002, min(0.01, np.percentile(np.abs(corr2_orig), 95) * 2))
        ax_in.set_xlim(-lim, lim)
        ax_in.set_ylim(-lim, lim)
        ax_in.set_title("zoom", fontsize=8)
        ax_in.tick_params(labelsize=7)
        ax_in.grid(alpha=0.3)
    except Exception:
        pass

    # --- (b) J distribution: full width ---
    ax_j = fig.add_subplot(gs[1, :])
    ax_j.hist(J_params, bins=min(50, max(10, len(J_params) // 5)), density=True,
              color="steelblue", edgecolor="white", alpha=0.85)
    ax_j.set_xlabel(r"coupling $J$")
    ax_j.set_ylabel(r"$P(J)$")
    ax_j.set_title("Distribution of pairwise couplings in multipliers")
    ax_j.grid(alpha=0.3)
    ax_j.text(0.02, 0.98, "(b)", transform=ax_j.transAxes, fontsize=12, fontweight="bold",
              va="top", ha="left")

    # --- (c) Triplet correlations ---
    ax_trip = fig.add_subplot(gs[2, 0])
    ax_trip.scatter(corr3_orig, corr3_model, s=10, alpha=0.45, c="darkred", edgecolors="none")
    lo3 = float(min(corr3_orig.min(), corr3_model.min()))
    hi3 = float(max(corr3_orig.max(), corr3_model.max()))
    ax_trip.plot([lo3, hi3], [lo3, hi3], "k--", lw=1)
    ax_trip.set_xlabel(r"measured $\langle \sigma_i \sigma_j \sigma_k \rangle$")
    ax_trip.set_ylabel(r"predicted $\langle \sigma_i \sigma_j \sigma_k \rangle$")
    ax_trip.set_title("Triplet correlation ($k=3$)")
    ax_trip.grid(alpha=0.3)
    ax_trip.text(0.02, 0.98, "(c)", transform=ax_trip.transAxes, fontsize=12, fontweight="bold",
                 va="top", ha="left")

    # --- (d) P(K): data vs Ising vs independent (Poisson-binomial from empirical p_i) ---
    ax_pk = fig.add_subplot(gs[2, 1])

    def _binary_spike_matrix(X):
        X = np.asarray(X, dtype=float)
        if X.min() < 0:  # {-1, +1}
            return (X + 1.0) / 2.0
        return (X > 0).astype(float)  # hard binarize

    def _spike_counts_per_bin(X):
        sigma = _binary_spike_matrix(X)
        return np.sum(sigma, axis=1).astype(int)

    sigma_data = _binary_spike_matrix(original_data)
    K_data = _spike_counts_per_bin(original_data)
    K_model = _spike_counts_per_bin(model_samples)

    # Independent model: p_i = ⟨σᵢ⟩ from data; P_ind(K) from ∏ᵢ [(1-p_i) + p_i z] via convolution.
    p_i_marginal = np.mean(sigma_data, axis=0)
    P_indep = np.array([1.0])
    for p_i in p_i_marginal:
        P_indep = np.convolve(P_indep, [1.0 - p_i, p_i])

    k_axis = np.arange(N + 1)
    P_data = np.bincount(K_data, minlength=N + 1).astype(float) / max(len(K_data), 1)
    P_ising = np.bincount(K_model, minlength=N + 1).astype(float) / max(len(K_model), 1)

    sigma = (original_data + 1) / 2  # should be {0,1}
    print("Unique values after conversion:", np.unique(sigma))
    print("K=0 count:", np.sum(sigma.sum(axis=1) == 0))
    print("K=1 count:", np.sum(sigma.sum(axis=1) == 1))
    print("Mean K:", sigma.sum(axis=1).mean())

    eps = 1e-12
    ax_pk.semilogy(k_axis, np.maximum(P_data, eps), "o-", color="blue", ms=4, lw=1.2, label="data")
    ax_pk.semilogy(k_axis, np.maximum(P_ising, eps), "o-", color="red", ms=4, lw=1.2, label="Ising")
    ax_pk.semilogy(k_axis, np.maximum(P_indep, eps), "-", color="black", lw=1.5,
                   label="independent")
    ax_pk.set_xlabel(r"$K$ (simultaneous spikes per bin)")
    ax_pk.set_ylabel(r"$P(K)$")
    ax_pk.set_title(
        r"$P(K)$: data vs fitted Ising vs independent Bernoullis "
        r"($p_i=\langle\sigma_i\rangle_{\mathrm{data}}$)"
    )
    ax_pk.legend(loc="upper right", fontsize=8)
    ax_pk.grid(alpha=0.3)
    k_hi = float(N if max_k_plot is None else min(N, max_k_plot))
    ax_pk.set_xlim(-0.5, k_hi + 0.5)
    ax_pk.text(0.02, 0.98, "(d)", transform=ax_pk.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    n_data = len(original_data)
    n_ising = len(model_samples)
    count_data = np.bincount(K_data, minlength=N + 1)
    count_ising = np.bincount(K_model, minlength=N + 1)

    # Save summary CSV
    summary_df = pd.DataFrame({
        "pairwise_corr_orig": corr2_orig,
        "pairwise_corr_model": corr2_model,
    })
    summary_df.to_csv(os.path.join(output_dir, "model_quality_summary_pairwise.csv"), index=False)

    trip_df = pd.DataFrame({
        "triplet_corr_orig": corr3_orig,
        "triplet_corr_model": corr3_model,
    })
    trip_df.to_csv(os.path.join(output_dir, "model_quality_summary_triplet.csv"), index=False)

    pk_df = pd.DataFrame({
        "K": k_axis,
        "count_data": count_data,
        "count_ising": count_ising,
        "P_data": P_data,
        "P_ising": P_ising,
        "P_independent": P_indep,
    })
    pk_df.to_csv(os.path.join(output_dir, "model_quality_summary_P_K.csv"), index=False)

    pd.DataFrame([{
        "N_neurons": N,
        "n_data_samples": n_data,
        "n_ising_samples": n_ising,
    }]).to_csv(os.path.join(output_dir, "model_quality_summary_P_K_metadata.csv"), index=False)

    pd.DataFrame({
        "neuron_index_1based": np.arange(1, N + 1),
        "p_i_marginal_data": p_i_marginal,
    }).to_csv(os.path.join(output_dir, "model_quality_summary_independent_marginals.csv"), index=False)

    pd.DataFrame({
        "neuron_index_1based": np.arange(1, N + 1),
        "h": h_params,
    }).to_csv(os.path.join(output_dir, "model_quality_summary_h_fields.csv"), index=False)

    j_rows = [
        {"neuron_i_1based": i + 1, "neuron_j_1based": j + 1, "J": J_params[idx]}
        for idx, (i, j) in enumerate(combinations(range(N), 2))
    ]
    pd.DataFrame(j_rows).to_csv(os.path.join(output_dir, "model_quality_summary_J_couplings.csv"), index=False)

    pd.DataFrame({"J": J_params}).to_csv(os.path.join(output_dir, "model_quality_summary_J_values.csv"), index=False)

    print(f"Model quality summary figure saved to {out_path}")


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

def plot_energy_distribution_by_k(results, output_dir, num_bins=50):
    """
    Plot a heatmap showing the distribution of energy values for each k value.
    
    Parameters:
    -----------
    results : dict
        Results from calculate_energy_by_neuron_count containing:
        - 'k_values': list of k values (number of neurons on)
        - 'energies_by_k': list of lists, where energies_by_k[i] contains energies for k=k_values[i]
    output_dir : str
        Directory to save plots
    num_bins : int, optional
        Number of bins for energy histogram (default=50)
        
    Returns:
    --------
    None
    """
    k_values = results['k_values']
    energies_by_k = results['energies_by_k']
    
    # Find the energy range across all k values
    all_energies = []
    for energies in energies_by_k:
        all_energies.extend(energies)
    
    if len(all_energies) == 0:
        print("Warning: No energy values to plot")
        return
    
    energy_min = np.min(all_energies)
    energy_max = np.max(all_energies)
    
    # Create bins for energy
    energy_bins = np.linspace(energy_min, energy_max, num_bins + 1)
    energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    
    # Create histogram matrix: rows = k values, columns = energy bins
    histogram_matrix = np.zeros((len(k_values), num_bins))
    
    for i, (k, energies) in enumerate(zip(k_values, energies_by_k)):
        if len(energies) > 0:
            hist, _ = np.histogram(energies, bins=energy_bins)
            histogram_matrix[i, :] = hist
    
    # Normalize each row to show probability density (optional - can also show counts)
    # For better visualization, normalize by max count per row
    row_maxes = histogram_matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1  # Avoid division by zero
    histogram_matrix_normalized = histogram_matrix / row_maxes
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use imshow for heatmap
    im = ax.imshow(histogram_matrix_normalized, aspect='auto', cmap='viridis', 
                   origin='lower', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xlabel('Energy Bin', fontsize=12)
    ax.set_ylabel('Number of Neurons On (k)', fontsize=12)
    ax.set_title('Energy Distribution Heatmap by Number of Neurons On', fontsize=14, fontweight='bold')
    
    # Set x-axis ticks (show fewer ticks for readability)
    n_x_ticks = min(10, num_bins)
    x_tick_indices = np.linspace(0, num_bins - 1, n_x_ticks).astype(int)
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([f'{energy_centers[idx]:.2f}' for idx in x_tick_indices], rotation=45)
    
    # Set y-axis ticks (show all k values)
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels(k_values)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', fontsize=11)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_distribution_by_k_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a version with raw counts (not normalized)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    im2 = ax2.imshow(histogram_matrix, aspect='auto', cmap='viridis', 
                     origin='lower', interpolation='nearest')
    
    ax2.set_xlabel('Energy Bin', fontsize=12)
    ax2.set_ylabel('Number of Neurons On (k)', fontsize=12)
    ax2.set_title('Energy Distribution Heatmap by Number of Neurons On (Raw Counts)', 
                  fontsize=14, fontweight='bold')
    
    ax2.set_xticks(x_tick_indices)
    ax2.set_xticklabels([f'{energy_centers[idx]:.2f}' for idx in x_tick_indices], rotation=45)
    ax2.set_yticks(range(len(k_values)))
    ax2.set_yticklabels(k_values)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Count', fontsize=11)
    
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_distribution_by_k_heatmap_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    heatmap_df = pd.DataFrame(histogram_matrix, 
                             index=[f'k={k}' for k in k_values],
                             columns=[f'E_{i:.2f}' for i in energy_centers])
    heatmap_df.to_csv(os.path.join(output_dir, "energy_distribution_by_k_heatmap.csv"))
    
    # Save summary statistics
    summary_data = []
    for k, energies in zip(k_values, energies_by_k):
        if len(energies) > 0:
            summary_data.append({
                'k': k,
                'mean_energy': np.mean(energies),
                'std_energy': np.std(energies),
                'min_energy': np.min(energies),
                'max_energy': np.max(energies),
                'median_energy': np.median(energies),
                'num_samples': len(energies)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "energy_by_k_summary.csv"), index=False)
    
    print(f"Energy distribution heatmap and CSVs saved to {output_dir}")