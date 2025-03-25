#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for the analysis of Purkinje neuron spike configurations.
This module includes functions for data loading, preprocessing, and general utilities.
"""

import numpy as np
import scipy
import scipy.io as spio
from scipy.interpolate import InterpolatedUnivariateSpline

def loadmat(filename):
    """
    Load MATLAB .mat files properly into Python dictionaries.
    This function handles the conversion of MATLAB structs to nested dictionaries.
    
    Parameters:
    -----------
    filename : str
        Path to the .mat file to load
        
    Returns:
    --------
    dict
        A dictionary containing the MATLAB data with proper Python types
    """
    def _check_keys(d):
        """
        Check if entries in dictionary are mat-objects. If yes,
        convert them to nested dictionaries.
        """
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs nested dictionaries from matobjects.
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def preprocessingSpikes(spikes, binSize):
    """
    Preprocess spike data by applying a sliding window and thresholding.
    
    Parameters:
    -----------
    spikes : ndarray
        Raw spike data
    binSize : int
        Size of the sliding window
        
    Returns:
    --------
    ndarray
        Convolved spike data (binary values)
    """
    convolved_spikes = 1 * (np.lib.stride_tricks.sliding_window_view(spikes, binSize, axis=0).sum(axis=2) > 0)
    return convolved_spikes

def checkPerturbationRegime(spikes, binsize):
    """
    Check perturbation regime for the spike data.
    
    Parameters:
    -----------
    spikes : ndarray
        Raw spike data
    binsize : int
        Size of the bins for preprocessing
        
    Returns:
    --------
    float
        N_c value representing the perturbation regime
    """
    # Run convolution
    spike_bins = preprocessingSpikes(spikes, binsize)

    # Count average spike per neuron & average of expected neuron fire rate 
    avgSpikePerNeuron = np.sum(spike_bins, axis=0) / spike_bins.shape[0]
    avgExpectedNeuronFire = np.sum(avgSpikePerNeuron) / spike_bins.shape[1]

    # Return n_c
    N_c = 1 / (avgExpectedNeuronFire * binsize)
    return N_c

def binSizeScan(spikes, start=1, end=200):
    """
    Scan over different bin sizes to determine optimal binning.
    
    Parameters:
    -----------
    spikes : ndarray
        Raw spike data
    start : int, optional
        Starting bin size (default=1)
    end : int, optional
        Ending bin size (default=200)
        
    Returns:
    --------
    ndarray
        Array of N_c values for each bin size
    """
    # Scan over bin sizes
    N_cList = []
    for binSize in range(start, end):
        N_cList += [checkPerturbationRegime(spikes, binSize)]

    return np.asarray(N_cList)

def Q_distribution(neuronsSpikeSet, numBins=100, label=""):
    """
    Calculate the distribution of the fraction of neurons that fired.
    
    Parameters:
    -----------
    neuronsSpikeSet : ndarray
        Spike data
    numBins : int, optional
        Number of bins for the histogram (default=100)
    label : str, optional
        Label for the plot (default="")
        
    Returns:
    --------
    tuple
        Histogram data as returned by plt.hist
    """
    import matplotlib.pyplot as plt
    
    # Initial params 
    numN = neuronsSpikeSet.shape[1]
    Q_samples = []
    neuronsSpikeSet = neuronsSpikeSet.copy()

    # Check data is on in [0, 1] scale, not [-1, 1]
    if np.min(neuronsSpikeSet) == -1:
        neuronsSpikeSet[neuronsSpikeSet == -1] = 0

    for n in neuronsSpikeSet:
        # Fraction of neurons that fired
        Q = np.sum(n) / (numN * 2.) + 1/2.
        Q_samples += [Q]
    
    # Calculate weight
    Q_samples = np.asarray(Q_samples)
    weight = np.ones_like(Q_samples) / np.prod(Q_samples.shape) 

    # Plot distributions 
    return plt.hist(Q_samples, bins=numBins, weights=weight, alpha=0.5, label=label, density=True)

def velocity_to_acceleration(timesteps, velocity):
    """
    Convert velocity data to acceleration using spline interpolation.
    
    Parameters:
    -----------
    timesteps : ndarray
        Time points
    velocity : ndarray
        Velocity data
        
    Returns:
    --------
    ndarray
        Acceleration data
    """
    sp = InterpolatedUnivariateSpline(timesteps, velocity)
    return sp.derivative()(timesteps)

def load_and_preprocess_data(matlab_file, truncate_idx_l=100, truncate_idx=800):
    """
    Load and preprocess data from a MATLAB file.
    
    Parameters:
    -----------
    matlab_file : str
        Path to the MATLAB file
    truncate_idx_l : int, optional
        Lower truncation index (default=100)
    truncate_idx : int, optional
        Upper truncation index (default=800)
        
    Returns:
    --------
    tuple
        (neural_stim, continous_stim) containing preprocessed data
    """
    b = loadmat(matlab_file)
    
    neural_stim = []
    continous_stim = []

    for i in b['kinAggrogate'].keys():
        print(f"Processing {i}")
        stim = b['kinAggrogate'][i]

        neural_session = []
        continous_sessions = []

        for k in stim.keys():
            print(f"\t Organizing {k}")
            dataMatrix = np.array(stim[k])

            if k[0] == "n":
                neural_session += [dataMatrix[truncate_idx_l:truncate_idx, 1:]]

            if k[0:2] == "l_":
                x = velocity_to_acceleration(dataMatrix[truncate_idx_l:truncate_idx, 0], 
                                           dataMatrix[truncate_idx_l:truncate_idx, -3]).reshape(-1, 1)
                y = velocity_to_acceleration(dataMatrix[truncate_idx_l:truncate_idx, 0], 
                                           dataMatrix[truncate_idx_l:truncate_idx, -2]).reshape(-1, 1)
                z = velocity_to_acceleration(dataMatrix[truncate_idx_l:truncate_idx, 0], 
                                           dataMatrix[truncate_idx_l:truncate_idx, -1]).reshape(-1, 1)

                dataMatrix = np.hstack([dataMatrix[truncate_idx_l:truncate_idx], x, y, z])
                continous_sessions += [dataMatrix[:, 1:]]

        neural_stim += [neural_session]
        continous_stim += [continous_sessions]
        
    return neural_stim, continous_stim

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate mean and confidence interval for a dataset.
    
    Parameters:
    -----------
    data : ndarray
        Input data
    confidence : float, optional
        Confidence level (default=0.95)
        
    Returns:
    --------
    tuple
        (mean, lower_bound, upper_bound)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def calculate_statistics_with_ci(data, axis=1, confidence=0.8):
    """
    Calculate statistics with confidence intervals for a dataset.
    
    Parameters:
    -----------
    data : ndarray
        Input data
    axis : int, optional
        Axis along which to calculate statistics (default=1)
    confidence : float, optional
        Confidence level (default=0.8)
        
    Returns:
    --------
    tuple
        (mean_values, lower_bounds, upper_bounds)
    """
    mean_values, lower_bounds, upper_bounds = [], [], []
    
    for i in range(data.shape[axis]):
        if axis == 1:
            m, ml, mu = mean_confidence_interval(data[:, i], confidence)
        else:
            m, ml, mu = mean_confidence_interval(data[i, :], confidence)
            
        mean_values.append(m)
        lower_bounds.append(ml)
        upper_bounds.append(mu)
        
    return mean_values, lower_bounds, upper_bounds

def calculate_time_dependent_firing_rate(spikes, window_size=10):
    """
    Calculate time-dependent firing rate using a sliding window.
    
    Parameters:
    -----------
    spikes : ndarray
        Binary spike data (0 or 1) with shape (time_bins, num_neurons)
    window_size : int, optional
        Size of the sliding window for rate calculation (default=10)
        
    Returns:
    --------
    ndarray
        Time-dependent firing rate averaged across neurons
    """
    # Ensure spikes are binary
    if np.min(spikes) < 0:
        spikes = (spikes > 0).astype(int)
    
    # Calculate firing rate using sliding window
    firing_rate = np.zeros(spikes.shape[0], dtype=float)
    half_window = window_size // 2
    
    for t in range(spikes.shape[0]):
        # Define window boundaries
        start_idx = max(0, t - half_window)
        end_idx = min(spikes.shape[0], t + half_window + 1)
        
        # Calculate rate in window for each neuron
        window_spikes = spikes[start_idx:end_idx]
        # Average across neurons
        firing_rate[t] = np.mean(window_spikes)
    
    return firing_rate