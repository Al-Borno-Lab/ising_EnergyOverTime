#!/usr/bin/env python
# coding: utf-8

"""
Analysis functions for the Ising model results.
This module contains functions for analyzing energy patterns and correlations.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from coniii.utils import k_corr
import matplotlib.pyplot as plt

def create_energy_spline(temp_range, energy_values):
    """
    Create a cubic spline interpolation of energy vs temperature.
    
    Parameters:
    -----------
    temp_range : ndarray
        Array of temperature values
    energy_values : ndarray
        Array of energy values corresponding to temperatures
        
    Returns:
    --------
    CubicSpline
        Spline function that maps temperature to energy
    """
    return CubicSpline(temp_range, energy_values)

def evaluate_model_quality(original_data, model_samples, max_corr_order=6):
    """
    Evaluate the quality of the fitted model by comparing correlations.
    
    Parameters:
    -----------
    original_data : ndarray
        Original spike data
    model_samples : ndarray
        Samples generated from the model
    max_corr_order : int, optional
        Maximum correlation order to evaluate (default=6)
        
    Returns:
    --------
    dict
        Dictionary of correlation values for each order
    """
    correlation_results = {}
    
    for i in range(2, max_corr_order + 1):
        original_corr = k_corr(original_data, i)
        model_corr = k_corr(model_samples, i)
        
        # Calculate correlation coefficient between original and model correlations
        correlation_results[i] = np.corrcoef(original_corr, model_corr)[0, 1]
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(original_corr, model_corr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f"{i}-point Correlation Comparison")
        plt.xlabel("Original Data Correlation")
        plt.ylabel("Model Correlation")
        plt.savefig(f"correlation_order_{i}.png")
        plt.close()
    
    return correlation_results

def analyze_neural_stimuli(neural_stim, continous_stim, multipliers, critical_energy, energy_temp_spline=None):
    """
    Analyze neural stimuli data with respect to the Ising model.
    
    Parameters:
    -----------
    neural_stim : list
        List of neural stimulation data
    continous_stim : list
        List of continuous stimulation data
    multipliers : ndarray
        Model parameters (h, J)
    critical_energy : float
        Critical energy from phase transition analysis
    energy_temp_spline : CubicSpline, optional
        Spline function mapping temperature to energy
        
    Returns:
    --------
    dict
        Dictionary of analysis results
    """
    from model import calc_e
    
    # Convert stimuli to numpy arrays if needed
    x_stim_0 = np.array([continous_stim[0][i][:, 0] for i in range(0, len(continous_stim[0]))])
    x_stim_1 = np.array([continous_stim[1][i][:, 0] for i in range(0, len(continous_stim[1]))])
    x_stim_2 = np.array([continous_stim[2][i][:, 0] for i in range(0, len(continous_stim[2]))])

    # Convert neural data to binary format
    neural_0 = (np.asarray([neural_stim[0][i][:, :] for i in range(0, len(neural_stim[0]))]) > 0) * 1
    neural_1 = (np.asarray([neural_stim[1][i][:, :] for i in range(0, len(neural_stim[1]))]) > 0) * 1
    neural_2 = (np.asarray([neural_stim[2][i][:, :] for i in range(0, len(neural_stim[2]))]) > 0) * 1

    # Calculate energy for each neural state
    e_0 = np.asarray([calc_e(i, multipliers) for i in neural_0])
    e_1 = np.asarray([calc_e(i, multipliers) for i in neural_1])
    e_2 = np.asarray([calc_e(i, multipliers) for i in neural_2])
    
    # Return comprehensive analysis data
    results = {
        'x_stim_data': [x_stim_0, x_stim_1, x_stim_2],
        'neural_binary': [neural_0, neural_1, neural_2],
        'energy_values': [e_0, e_1, e_2],
        'critical_energy': critical_energy
    }
    
    # If spline function is provided, calculate effective temperatures
    if energy_temp_spline is not None:
        # This would map energy back to temperature for interpretation
        # Note: This is an approximation and may not be valid for all energy values
        # We would need to handle values outside the interpolation range
        pass
    
    return results

def calculate_statistics_across_trials(analysis_results, confidence=0.8):
    """
    Calculate statistics across trials for both kinematic and neural data.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from the analyze_neural_stimuli function
    confidence : float, optional
        Confidence level for intervals (default=0.8)
        
    Returns:
    --------
    dict
        Dictionary containing mean and confidence intervals for each measure
    """
    from utils import mean_confidence_interval
    
    x_stim_data = analysis_results['x_stim_data']
    energy_values = analysis_results['energy_values']
    
    # Initialize containers for results
    kinematics_stats = []
    energy_stats = []
    
    # Calculate statistics for each stimulation condition
    for i, (x_stim, e_vals) in enumerate(zip(x_stim_data, energy_values)):
        # Kinematics statistics
        kin_mean, kin_lower, kin_upper = [], [], []
        for j in range(x_stim.shape[1]):
            m, ml, mu = mean_confidence_interval(x_stim[:, j], confidence)
            kin_mean.append(m)
            kin_lower.append(ml)
            kin_upper.append(mu)
        
        # Energy statistics
        energy_mean, energy_lower, energy_upper = [], [], []
        for j in range(e_vals.shape[1]):
            m, ml, mu = mean_confidence_interval(e_vals[:, j], confidence)
            energy_mean.append(m)
            energy_lower.append(ml)
            energy_upper.append(mu)
        
        kinematics_stats.append({
            'mean': kin_mean,
            'lower': kin_lower,
            'upper': kin_upper
        })
        
        energy_stats.append({
            'mean': energy_mean,
            'lower': energy_lower,
            'upper': energy_upper
        })
    
    return {
        'kinematics': kinematics_stats,
        'energy': energy_stats
    }

def identify_transition_points(energy_data, kinematic_data, threshold=0.2):
    """
    Identify potential transition points in neural activity based on energy changes.
    
    Parameters:
    -----------
    energy_data : list or ndarray
        Energy values over time
    kinematic_data : list or ndarray
        Kinematic values over time
    threshold : float, optional
        Threshold for identifying significant changes (default=0.2)
        
    Returns:
    --------
    list
        Indices of potential transition points
    """
    # Convert to numpy arrays if needed
    energy_data = np.array(energy_data)
    kinematic_data = np.array(kinematic_data)
    
    # Calculate derivatives
    energy_deriv = np.gradient(energy_data)
    kinematic_deriv = np.gradient(kinematic_data)
    
    # Find points where energy derivative exceeds threshold
    significant_points = np.where(np.abs(energy_deriv) > threshold * np.std(energy_deriv))[0]
    
    # Filter points to find those with corresponding kinematic changes
    transition_points = []
    for point in significant_points:
        if point > 0 and point < len(kinematic_deriv) - 1:
            if np.abs(kinematic_deriv[point]) > np.std(kinematic_deriv):
                transition_points.append(point)
    
    return transition_points