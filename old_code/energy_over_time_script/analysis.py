#!/usr/bin/env python
# coding: utf-8

"""
Analysis functions for the Ising model results.
This module contains functions for analyzing energy patterns and correlations.
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
#from coniii.utils import k_corr
import matplotlib.pyplot as plt
from utils import calculate_time_dependent_firing_rate

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

def analyze_neural_stimuli(neural_stim, continous_stim, multipliers, critical_energy, fr_window=10, energy_temp_spline=None, output_dir=None):
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
    output_dir : str, optional
        Directory to save analysis results
        
    Returns:
    --------
    dict
        Dictionary of analysis results
    """
    from model import calc_e_with_terms
    
    # Convert stimuli to numpy arrays if needed
    x_stim_0 = np.array([continous_stim[0][i][:, 0] for i in range(0, len(continous_stim[0]))])
    x_stim_1 = np.array([continous_stim[1][i][:, 0] for i in range(0, len(continous_stim[1]))])
    x_stim_2 = np.array([continous_stim[2][i][:, 0] for i in range(0, len(continous_stim[2]))])

    # Extract y and z coordinates as well
    y_stim_0 = np.array([continous_stim[0][i][:, 1] for i in range(0, len(continous_stim[0]))])
    y_stim_1 = np.array([continous_stim[1][i][:, 1] for i in range(0, len(continous_stim[1]))])
    y_stim_2 = np.array([continous_stim[2][i][:, 1] for i in range(0, len(continous_stim[2]))])

    z_stim_0 = np.array([continous_stim[0][i][:, 2] for i in range(0, len(continous_stim[0]))])
    z_stim_1 = np.array([continous_stim[1][i][:, 2] for i in range(0, len(continous_stim[1]))])
    z_stim_2 = np.array([continous_stim[2][i][:, 2] for i in range(0, len(continous_stim[2]))])

    # Convert neural data to binary format {0, 1}
    neural_0 = (np.asarray([neural_stim[0][i][:, :] for i in range(0, len(neural_stim[0]))]) > 0) * 1
    neural_1 = (np.asarray([neural_stim[1][i][:, :] for i in range(0, len(neural_stim[1]))]) > 0) * 1
    neural_2 = (np.asarray([neural_stim[2][i][:, :] for i in range(0, len(neural_stim[2]))]) > 0) * 1

    # Ising spins {-1, +1} for energy (matches model fitting / calc_e convention)
    neural_0_ising = 2 * neural_0 - 1
    neural_1_ising = 2 * neural_1 - 1
    neural_2_ising = 2 * neural_2 - 1

    # Calculate energy for each neural state
    # returns both terms e, j, h
    e_0 = np.asarray([calc_e_with_terms(i, multipliers) for i in neural_0_ising])
    e_1 = np.asarray([calc_e_with_terms(i, multipliers) for i in neural_1_ising])
    e_2 = np.asarray([calc_e_with_terms(i, multipliers) for i in neural_2_ising])

    # pull out hamiltonian terms  
    j_0 = e_0[:, 1]
    h_0 = e_0[:, 2]
    j_1 = e_1[:, 1]
    h_1 = e_1[:, 2]
    j_2 = e_2[:, 1]
    h_2 = e_2[:, 2] 

    # pull out energy terms
    e_0 = e_0[:, 0]
    e_1 = e_1[:, 0]
    e_2 = e_2[:, 0]

    # firing rate
    f_0 = [np.asarray(calculate_time_dependent_firing_rate(n_0, window_size=fr_window)) for  n_0 in neural_0]
    f_1 = [np.asarray(calculate_time_dependent_firing_rate(n_1, window_size=fr_window)) for n_1 in neural_1] 
    f_2 = [np.asarray(calculate_time_dependent_firing_rate(n_2, window_size=fr_window)) for n_2 in neural_2]
    
    # Create reach_idx arrays for each stimulus (each may have different number of trials)
    reach_idx_0 = [[i]*len(v) for i, v in enumerate(continous_stim[0])]
    reach_idx_1 = [[i]*len(v) for i, v in enumerate(continous_stim[1])]
    reach_idx_2 = [[i]*len(v) for i, v in enumerate(continous_stim[2])]

    # Return comprehensive analysis data
    results = {
        'x_stim_data': [x_stim_0, x_stim_1, x_stim_2],
        'y_stim_data': [y_stim_0, y_stim_1, y_stim_2],
        'z_stim_data': [z_stim_0, z_stim_1, z_stim_2],
        'neural_binary': [neural_0, neural_1, neural_2],
        'energy_values': [e_0, e_1, e_2],
        'firing_rate_values': [f_0, f_1, f_2],
        'j_values': [j_0, j_1, j_2],
        'h_values': [h_0, h_1, h_2],
        'critical_energy': critical_energy
    }
    
    # If output directory is provided, save additional analysis data
    if output_dir:
        # Save neural stimuli data summary
        stim_summary = {
            'Stim_Index': range(3),
            'Stim_Type': ['Stim_0', 'Stim_1', 'Stim_2'],
            'Num_Trials': [neural_0.shape[0], neural_1.shape[0], neural_2.shape[0]],
            'Num_Timepoints': [neural_0.shape[1], neural_1.shape[1], neural_2.shape[1]],
            'Num_Neurons': [neural_0.shape[2], neural_1.shape[2], neural_2.shape[2]],
            'Mean_Energy': [np.mean(e_0), np.mean(e_1), np.mean(e_2)],
            'Min_Energy': [np.min(e_0), np.min(e_1), np.min(e_2)],
            'Max_Energy': [np.max(e_0), np.max(e_1), np.max(e_2)],
            'Mean_J': [np.mean(j_0), np.mean(j_1), np.mean(j_2)],
            'Min_J': [np.min(j_0), np.min(j_1), np.min(j_2)],
            'Max_J': [np.max(j_0), np.max(j_1), np.max(j_2)],
            'Mean_H': [np.mean(h_0), np.mean(h_1), np.mean(h_2)],
            'Min_H': [np.min(h_0), np.min(h_1), np.min(h_2)],
            'Max_H': [np.max(h_0), np.max(h_1), np.max(h_2)]
        }
        pd.DataFrame(stim_summary).to_csv(os.path.join(output_dir, "neural_stimuli_summary.csv"), index=False)
    
        # output all reaches
        all_reaches = {
            "reach_idx": [],
            "stim":[],
            "x":[],
            "y":[],
            "z":[],
            "firing_rate": [],
            "energy": [],
            "j": [],
            "h": []
        }
        
        for i in range(len(continous_stim[0])):
            all_reaches["reach_idx"] += reach_idx_0[i]
            all_reaches["stim"] += [0] * len(reach_idx_0[i])
            all_reaches["x"] += x_stim_0[i].tolist()
            all_reaches["y"] += y_stim_0[i].tolist()
            all_reaches["z"] += z_stim_0[i].tolist()
            all_reaches["firing_rate"] += f_0[i].tolist()
            all_reaches["energy"]+= e_0[i].tolist()
            all_reaches["j"] += j_0[i].tolist()
            all_reaches["h"] += h_0[i].tolist()
        
        for i in range(len(continous_stim[1])):
            all_reaches["reach_idx"]+= reach_idx_1[i]
            all_reaches["stim"] += [1] * len(reach_idx_1[i])
            all_reaches["x"] += x_stim_1[i].tolist()
            all_reaches["y"] += y_stim_1[i].tolist()
            all_reaches["z"] += z_stim_1[i].tolist()
            all_reaches["firing_rate"] += f_1[i].tolist()
            all_reaches["energy"] += e_1[i].tolist()
            all_reaches["j"] += j_1[i].tolist()
            all_reaches["h"] += h_1[i].tolist()

        for i in range(len(continous_stim[2])):
            all_reaches["reach_idx"] += reach_idx_2[i]
            all_reaches["stim"] += [2] * len(reach_idx_2[i])
            all_reaches["x"] += x_stim_2[i].tolist()
            all_reaches["y"] += y_stim_2[i].tolist()
            all_reaches["z"] += z_stim_2[i].tolist()
            all_reaches["firing_rate"] += f_2[i].tolist()
            all_reaches["energy"] += e_2[i].tolist()
            all_reaches["j"] += j_2[i].tolist()
            all_reaches["h"] += h_2[i].tolist()

        pd.DataFrame(all_reaches).to_csv(os.path.join(output_dir, "per_reach_state.csv"), index=False)


    # If spline function is provided, calculate effective temperatures
    if energy_temp_spline is not None and output_dir:
        # Try to map energy back to temperature for interpretation
        # Note: This is an approximation and may not be valid for all energy values
        # We need to handle values outside the interpolation range
        try:
            # Get the valid range for the spline
            spline_x_min = energy_temp_spline.x[0]
            spline_x_max = energy_temp_spline.x[-1]
            
            # Function to map energy to temperature within valid range
            def energy_to_temp(energy):
                # Clip energy values to the valid range for the spline
                clipped_energy = np.clip(energy, spline_x_min, spline_x_max)
                # Find temperature values through inverse lookup
                temps = np.linspace(0.1, 2.0, 100)  # Temperature range
                spline_energies = energy_temp_spline(temps)
                # Find closest temperature for each energy
                return [temps[np.abs(spline_energies - e).argmin()] for e in clipped_energy]
            
            # Calculate effective temperature for mean energy of each stimulus
            mean_energies = [np.mean(e_0), np.mean(e_1), np.mean(e_2)]
            effective_temps = energy_to_temp(mean_energies)
            
            # Save to CSV
            temp_mapping = {
                'Stim_Type': ['Stim_0', 'Stim_1', 'Stim_2'],
                'Mean_Energy': mean_energies,
                'Effective_Temperature': effective_temps
            }
            pd.DataFrame(temp_mapping).to_csv(os.path.join(output_dir, "energy_temperature_mapping.csv"), index=False)
        except Exception as e:
            print(f"Warning: Could not map energy to temperature: {e}")
    
    return results


def analyze_single_stimulus(neural_stim_i, continous_stim_i, multipliers, critical_energy,
                             stim_idx=0, fr_window=10, energy_temp_spline=None, output_dir=None):
    """
    Run the full analysis pipeline for a *single* stimulus using its own Ising model.

    This mirrors the per-stimulus logic inside analyze_neural_stimuli but operates
    on one stimulus at a time so each stim can have its own fitted multipliers.

    Parameters
    ----------
    neural_stim_i    : list of arrays — spike data for one stimulus (one entry per reach)
    continous_stim_i : list of arrays — kinematic data for the same stimulus
    multipliers      : ndarray — Ising model parameters (h, J) fitted to THIS stimulus
    critical_energy  : float — critical energy from phase transition analysis
    stim_idx         : int — stimulus index used for labelling outputs (default 0)
    fr_window        : int — window size for time-dependent firing rate (default 10)
    energy_temp_spline : CubicSpline or None
    output_dir       : str or None — directory to save per_reach_state.csv and summary

    Returns
    -------
    dict compatible with calculate_statistics_across_trials (single-element lists)
    """
    from model import calc_e_with_terms

    x_s = np.array([continous_stim_i[j][:, 0] for j in range(len(continous_stim_i))])
    y_s = np.array([continous_stim_i[j][:, 1] for j in range(len(continous_stim_i))])
    z_s = np.array([continous_stim_i[j][:, 2] for j in range(len(continous_stim_i))])

    neural_s = (np.asarray([neural_stim_i[j][:, :] for j in range(len(neural_stim_i))]) > 0) * 1
    neural_s_ising = 2 * neural_s - 1

    e_s = np.asarray([calc_e_with_terms(k, multipliers) for k in neural_s_ising])
    j_s = e_s[:, 1]
    h_s = e_s[:, 2]
    e_s = e_s[:, 0]

    from utils import calculate_time_dependent_firing_rate
    f_s = [np.asarray(calculate_time_dependent_firing_rate(n, window_size=fr_window))
           for n in neural_s]

    reach_idx_s = [[j] * len(v) for j, v in enumerate(continous_stim_i)]

    # Wrap in single-element lists so calculate_statistics_across_trials works unchanged
    results = {
        'x_stim_data':       [x_s],
        'y_stim_data':       [y_s],
        'z_stim_data':       [z_s],
        'neural_binary':     [neural_s],
        'energy_values':     [e_s],
        'firing_rate_values': [f_s],
        'j_values':          [j_s],
        'h_values':          [h_s],
        'critical_energy':   critical_energy,
    }

    if output_dir:
        stim_label = f'Stim_{stim_idx}'
        stim_summary = {
            'Stim_Index':    [stim_idx],
            'Stim_Type':     [stim_label],
            'Num_Trials':    [neural_s.shape[0]],
            'Num_Timepoints': [neural_s.shape[1]],
            'Num_Neurons':   [neural_s.shape[2]],
            'Mean_Energy':   [np.mean(e_s)],
            'Min_Energy':    [np.min(e_s)],
            'Max_Energy':    [np.max(e_s)],
            'Mean_J':        [np.mean(j_s)],
            'Min_J':         [np.min(j_s)],
            'Max_J':         [np.max(j_s)],
            'Mean_H':        [np.mean(h_s)],
            'Min_H':         [np.min(h_s)],
            'Max_H':         [np.max(h_s)],
        }
        pd.DataFrame(stim_summary).to_csv(
            os.path.join(output_dir, "neural_stimuli_summary.csv"), index=False)

        n_neurons = neural_s.shape[2] if neural_s.ndim == 3 else 0
        spike_keys = [f"spike_n{k}" for k in range(n_neurons)]
        all_reaches = {
            "reach_idx": [], "stim": [],
            "x": [], "y": [], "z": [],
            "firing_rate": [], "energy": [], "j": [], "h": [],
            **{k: [] for k in spike_keys},
        }
        for i in range(len(continous_stim_i)):
            n_t = len(reach_idx_s[i])
            all_reaches["reach_idx"]   += reach_idx_s[i]
            all_reaches["stim"]        += [stim_idx] * n_t
            all_reaches["x"]           += x_s[i].tolist()
            all_reaches["y"]           += y_s[i].tolist()
            all_reaches["z"]           += z_s[i].tolist()
            all_reaches["firing_rate"] += f_s[i].tolist()
            all_reaches["energy"]      += e_s[i].tolist()
            all_reaches["j"]           += j_s[i].tolist()
            all_reaches["h"]           += h_s[i].tolist()
            # Per-neuron binary spikes (0/1)
            for k in range(n_neurons):
                all_reaches[f"spike_n{k}"] += neural_s[i, :n_t, k].tolist()
        pd.DataFrame(all_reaches).to_csv(
            os.path.join(output_dir, "per_reach_state.csv"), index=False)

        if energy_temp_spline is not None:
            try:
                spline_x_min = energy_temp_spline.x[0]
                spline_x_max = energy_temp_spline.x[-1]
                temps = np.linspace(0.1, 2.0, 100)
                spline_energies = energy_temp_spline(temps)
                mean_e = float(np.mean(e_s))
                clipped = np.clip(mean_e, spline_x_min, spline_x_max)
                eff_temp = temps[np.abs(spline_energies - clipped).argmin()]
                pd.DataFrame({
                    'Stim_Type':            [stim_label],
                    'Mean_Energy':          [mean_e],
                    'Effective_Temperature': [eff_temp],
                }).to_csv(os.path.join(output_dir, "energy_temperature_mapping.csv"), index=False)
            except Exception as ex:
                print(f"Warning: Could not map energy to temperature: {ex}")

    return results


def analyze_neural_stimuli_wells(neural_stim, continous_stim, multipliers, critical_energy, energy_temp_spline=None, output_dir=None):
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
    output_dir : str, optional
        Directory to save analysis results
        
    Returns:
    --------
    dict
        Dictionary of analysis results
    """
    from model import calc_e
    
    # Convert stimuli to numpy arrays if needed
    x_stim_0 = np.array([continous_stim[0][i][:, 0] for i in range(0, len(continous_stim[0]))])

    # Convert neural data to binary format
    neural_0 = (np.asarray([neural_stim[0][i][:, :] for i in range(0, len(neural_stim[0]))]) > 0) * 1

    # Calculate energy for each neural state
    e_0 = np.asarray([calc_e(i, multipliers) for i in neural_0])
    
    # Return comprehensive analysis data
    results = {
        'x_stim_data': [x_stim_0],
        'neural_binary': [neural_0,],
        'energy_values': [e_0],
        'critical_energy': critical_energy
    }
    
    # If output directory is provided, save additional analysis data
    if output_dir:
        # Save neural stimuli data summary
        stim_summary = {
            'Stim_Index': range(1),
            'Stim_Type': ['Stim_0'],
            'Num_Trials': [neural_0.shape[0]],
            'Num_Timepoints': [neural_0.shape[1]],
            'Num_Neurons': [neural_0.shape[2]],
            'Mean_Energy': [np.mean(e_0)],
            'Min_Energy': [np.min(e_0)],
            'Max_Energy': [np.max(e_0)]
        }
        pd.DataFrame(stim_summary).to_csv(os.path.join(output_dir, "neural_stimuli_summary.csv"), index=False)
    
    # If spline function is provided, calculate effective temperatures
    if energy_temp_spline is not None and output_dir:
        # Try to map energy back to temperature for interpretation
        # Note: This is an approximation and may not be valid for all energy values
        # We need to handle values outside the interpolation range
        try:
            # Get the valid range for the spline
            spline_x_min = energy_temp_spline.x[0]
            spline_x_max = energy_temp_spline.x[-1]
            
            # Function to map energy to temperature within valid range
            def energy_to_temp(energy):
                # Clip energy values to the valid range for the spline
                clipped_energy = np.clip(energy, spline_x_min, spline_x_max)
                # Find temperature values through inverse lookup
                temps = np.linspace(0.1, 2.0, 100)  # Temperature range
                spline_energies = energy_temp_spline(temps)
                # Find closest temperature for each energy
                return [temps[np.abs(spline_energies - e).argmin()] for e in clipped_energy]
            
            # Calculate effective temperature for mean energy of each stimulus
            mean_energies = [np.mean(e_0)]
            effective_temps = energy_to_temp(mean_energies)
            
            # Save to CSV
            temp_mapping = {
                'Stim_Type': ['Stim_0'],
                'Mean_Energy': mean_energies,
                'Effective_Temperature': effective_temps
            }
            pd.DataFrame(temp_mapping).to_csv(os.path.join(output_dir, "energy_temperature_mapping.csv"), index=False)
        except Exception as e:
            print(f"Warning: Could not map energy to temperature: {e}")
    
    return results


def calculate_statistics_across_trials(analysis_results, confidence=0.8, output_dir=None):
    """
    Calculate statistics across trials for kinematic (x, y, z) and neural data.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from the analyze_neural_stimuli function
    confidence : float, optional
        Confidence level for intervals (default=0.8)
    output_dir : str, optional
        Directory to save statistics results
        
    Returns:
    --------
    dict
        Dictionary containing mean and confidence intervals for each measure
    """
    from utils import mean_confidence_interval
    
    x_stim_data = analysis_results['x_stim_data']
    y_stim_data = analysis_results['y_stim_data']
    z_stim_data = analysis_results['z_stim_data']
    energy_values = analysis_results['energy_values']
    j_values = analysis_results['j_values']
    h_values = analysis_results['h_values']
    
    # Initialize containers for results
    x_kinematics_stats = []
    y_kinematics_stats = []
    z_kinematics_stats = []
    energy_stats = []
    j_values_stats = []
    h_values_stats = []
    
    # Calculate statistics for each stimulation condition
    for i, (x_stim, y_stim, z_stim, e_vals) in enumerate(zip(x_stim_data, y_stim_data, z_stim_data, energy_values)):
        # X-coordinate statistics
        x_mean, x_lower, x_upper = [], [], []
        for j in range(x_stim.shape[1]):
            m, ml, mu = mean_confidence_interval(x_stim[:, j], confidence)
            x_mean.append(m)
            x_lower.append(ml)
            x_upper.append(mu)
        
        # Y-coordinate statistics
        y_mean, y_lower, y_upper = [], [], []
        for j in range(y_stim.shape[1]):
            m, ml, mu = mean_confidence_interval(y_stim[:, j], confidence)
            y_mean.append(m)
            y_lower.append(ml)
            y_upper.append(mu)
        
        # Z-coordinate statistics
        z_mean, z_lower, z_upper = [], [], []
        for j in range(z_stim.shape[1]):
            m, ml, mu = mean_confidence_interval(z_stim[:, j], confidence)
            z_mean.append(m)
            z_lower.append(ml)
            z_upper.append(mu)
        
        # Energy statistics
        energy_mean, energy_lower, energy_upper = [], [], []
        for j in range(e_vals.shape[1]):
            m, ml, mu = mean_confidence_interval(e_vals[:, j], confidence)
            energy_mean.append(m)
            energy_lower.append(ml)
            energy_upper.append(mu)
        
        # J and H statistics
        j_mean, j_lower, j_upper = [], [], []
        h_mean, h_lower, h_upper = [], [], []
        for j in range(j_values[i].shape[1]):
            j_m, j_ml, j_mu = mean_confidence_interval(j_values[i][:, j], confidence)
            h_m, h_ml, h_mu = mean_confidence_interval(h_values[i][:, j], confidence)
            j_mean.append(j_m)
            j_lower.append(j_ml)
            j_upper.append(j_mu)
            h_mean.append(h_m)
            h_lower.append(h_ml)
            h_upper.append(h_mu)


        x_kinematics_stats.append({
            'mean': x_mean,
            'lower': x_lower,
            'upper': x_upper
        })
        
        y_kinematics_stats.append({
            'mean': y_mean,
            'lower': y_lower,
            'upper': y_upper
        })
        
        z_kinematics_stats.append({
            'mean': z_mean,
            'lower': z_lower,
            'upper': z_upper
        })
        
        energy_stats.append({
            'mean': energy_mean,
            'lower': energy_lower,
            'upper': energy_upper
        })
        
        j_values_stats.append({
            'mean': j_mean,
            'lower': j_lower,
            'upper': j_upper
        })
        h_values_stats.append({
            'mean': h_mean,
            'lower': h_lower,
            'upper': h_upper
        })

        # Save statistics to CSV and create plots if output directory is provided
        if output_dir:
            time_points = list(range(len(x_mean)))
            
            # Save X-coordinate statistics
            x_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': x_mean,
                'Lower_CI': x_lower,
                'Upper_CI': x_upper
            })
            x_stats_df.to_csv(os.path.join(output_dir, f"x_kinematics_stats_stim_{i}.csv"), index=False)
            
            # Save Y-coordinate statistics
            y_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': y_mean,
                'Lower_CI': y_lower,
                'Upper_CI': y_upper
            })
            y_stats_df.to_csv(os.path.join(output_dir, f"y_kinematics_stats_stim_{i}.csv"), index=False)
            
            # Save Z-coordinate statistics
            z_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': z_mean,
                'Lower_CI': z_lower,
                'Upper_CI': z_upper
            })
            z_stats_df.to_csv(os.path.join(output_dir, f"z_kinematics_stats_stim_{i}.csv"), index=False)
            
            # Save Energy statistics
            energy_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': energy_mean,
                'Lower_CI': energy_lower,
                'Upper_CI': energy_upper
            })
            energy_stats_df.to_csv(os.path.join(output_dir, f"energy_stats_stim_{i}.csv"), index=False)
            
            # Save J statistics
            j_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': j_mean,
                'Lower_CI': j_lower,
                'Upper_CI': j_upper
            })
            j_stats_df.to_csv(os.path.join(output_dir, f"j_stats_stim_{i}.csv"), index=False)
            
            # Save H statistics
            h_stats_df = pd.DataFrame({
                'Time': time_points,
                'Mean': h_mean,
                'Lower_CI': h_lower,
                'Upper_CI': h_upper
            })
            h_stats_df.to_csv(os.path.join(output_dir, f"h_stats_stim_{i}.csv"), index=False)


            # Create stacked plots for X, Y, Z coordinates
            fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            
            # X-coordinate plot
            axs[0].plot(time_points, x_mean, 'b-', linewidth=2, label='Mean X')
            axs[0].fill_between(time_points, x_lower, x_upper, alpha=0.3, color='blue', label=f'{int(confidence*100)}% CI')
            axs[0].set_ylabel('X Coordinate')
            axs[0].set_title(f'X-Coordinate Statistics - Stimulus {i}')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)
            
            # Y-coordinate plot
            axs[1].plot(time_points, y_mean, 'g-', linewidth=2, label='Mean Y')
            axs[1].fill_between(time_points, y_lower, y_upper, alpha=0.3, color='green', label=f'{int(confidence*100)}% CI')
            axs[1].set_ylabel('Y Coordinate')
            axs[1].set_title(f'Y-Coordinate Statistics - Stimulus {i}')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)
            
            # Z-coordinate plot
            axs[2].plot(time_points, z_mean, 'r-', linewidth=2, label='Mean Z')
            axs[2].fill_between(time_points, z_lower, z_upper, alpha=0.3, color='red', label=f'{int(confidence*100)}% CI')
            axs[2].set_ylabel('Z Coordinate')
            axs[2].set_title(f'Z-Coordinate Statistics - Stimulus {i}')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            # Energy plot
            axs[3].plot(time_points, energy_mean, 'purple', linewidth=2, label='Mean Energy')
            axs[3].fill_between(time_points, energy_lower, energy_upper, alpha=0.3, color='purple', label=f'{int(confidence*100)}% CI')
            axs[3].set_ylabel('Energy')
            axs[3].set_xlabel('Time Points')
            axs[3].set_title(f'Energy Statistics - Stimulus {i}')
            axs[3].legend()
            axs[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"stacked_statistics_stim_{i}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    return {
        'x_kinematics': x_kinematics_stats,
        'y_kinematics': y_kinematics_stats,
        'z_kinematics': z_kinematics_stats,
        'energy': energy_stats,
        'j_values': j_values_stats,
        'h_values': h_values_stats
    }

def identify_transition_points(energy_data, kinematic_data, threshold=0.2, output_dir=None, stim_idx=0):
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
    output_dir : str, optional
        Directory to save transition points data
    stim_idx : int, optional
        Stimulus index for file naming
        
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
    transition_points = significant_points
    # for point in significant_points:
    #     if point > 0 and point < len(kinematic_deriv) - 1:
    #         if np.abs(kinematic_deriv[point]) > np.std(kinematic_deriv):
    #             transition_points.append(point)
    
    # Save transition points data to CSV if output directory is provided
    if output_dir:
        # Create a dataframe with all analysis data
        transition_df = pd.DataFrame({
            'Time_Index': range(len(energy_data)),
            'Energy': energy_data,
            'Energy_Derivative': energy_deriv,
            'Kinematics': kinematic_data,
            'Kinematics_Derivative': kinematic_deriv,
            'Is_Transition_Point': [1 if i in transition_points else 0 for i in range(len(energy_data))]
        })
        transition_df.to_csv(os.path.join(output_dir, f"transition_analysis_stim_{stim_idx}.csv"), index=False)
    
    return transition_points


def identify_firing_rate_transition_points(firing_rate_data, kinematic_data, threshold=0.2, output_dir=None, stim_idx=0):
    """
    Identify potential transition points in neural activity based on firing rate changes.
    
    Uses the same derivative-based logic as identify_transition_points but applied to
    firing rate instead of energy. This allows comparison of which kinematic transitions
    are detected by firing rate vs energy.
    
    Parameters:
    -----------
    firing_rate_data : list or ndarray
        Firing rate values over time (mean across trials or single trial)
    kinematic_data : list or ndarray
        Kinematic values over time (position)
    threshold : float, optional
        Threshold for identifying significant changes (default=0.2)
    output_dir : str, optional
        Directory to save transition points data
    stim_idx : int, optional
        Stimulus index for file naming
        
    Returns:
    --------
    list
        Indices of potential transition points based on firing rate
    """
    firing_rate_data = np.array(firing_rate_data)
    kinematic_data = np.array(kinematic_data)
    
    # Ensure same length (firing rate may be shorter due to window)
    min_len = min(len(firing_rate_data), len(kinematic_data))
    firing_rate_data = firing_rate_data[:min_len]
    kinematic_data = kinematic_data[:min_len]
    
    # Calculate derivatives
    fr_deriv = np.gradient(firing_rate_data)
    kinematic_deriv = np.gradient(kinematic_data)
    
    # Find points where firing rate derivative exceeds threshold
    fr_std = np.std(fr_deriv)
    if fr_std > 0:
        significant_points = np.where(np.abs(fr_deriv) > threshold * fr_std)[0]
    else:
        significant_points = np.array([])
    
    # Filter points to find those with corresponding kinematic changes
    transition_points = []
    kin_std = np.std(kinematic_deriv)
    for point in significant_points:
        if point > 0 and point < len(kinematic_deriv) - 1 and kin_std > 0:
            if np.abs(kinematic_deriv[point]) > kin_std:
                transition_points.append(point)
    
    # Save transition points data to CSV if output directory is provided
    if output_dir:
        transition_df = pd.DataFrame({
            'Time_Index': range(len(firing_rate_data)),
            'Firing_Rate': firing_rate_data,
            'Firing_Rate_Derivative': fr_deriv,
            'Kinematics': kinematic_data,
            'Kinematics_Derivative': kinematic_deriv,
            'Is_Transition_Point': [1 if i in transition_points else 0 for i in range(len(firing_rate_data))]
        })
        transition_df.to_csv(os.path.join(output_dir, f"transition_analysis_firing_rate_stim_{stim_idx}.csv"), index=False)
    
    return transition_points
