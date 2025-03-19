#!/usr/bin/env python
# coding: utf-8

"""
Visualization functions for the Ising model analysis.
This module contains functions for creating and saving plots of model results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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
    None
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
    
    print(f"Phase transition plots saved to {output_dir}")
    
    return CubicSpline(temp_range, pos_avg_energy)

def plot_energy_across_time(stats, critical_energy, output_dir, title_prefix="", show_mid_point=True):
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
        
    Returns:
    --------
    None
    """
    kinematics = stats['kinematics']
    energy = stats['energy']
    
    for i, (kin, eng) in enumerate(zip(kinematics, energy)):
        plt.figure(figsize=(12, 10))
        
        # Top subplot for kinematics
        plt.subplot(2, 1, 1)
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
        
        # Bottom subplot for energy
        plt.subplot(2, 1, 2)
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"energy_kinematics_stim_{i}.png"))
        plt.close()
    
    print(f"Energy and kinematic plots saved to {output_dir}")

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
    
    print(f"Transition points plot saved to {output_dir}")

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
    
    print(f"Energy histogram saved to {output_dir}")

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
    
    print(f"Model quality plots saved to {output_dir}")