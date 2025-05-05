#!/usr/bin/env python
# coding: utf-8

"""
Ising model implementation for analyzing neural spike data.
This module contains functions for energy calculations, Metropolis sampling,
and phase transition analysis.
"""

import numpy as np
from numba import jit, njit
import time
from coniii import MCH

@njit(cache=True)
def fast_sum(J, s):
    """
    Helper function for calculating energy by iterating through couplings J.
    Optimized with numba for performance.
    
    Parameters:
    -----------
    J : ndarray
        Coupling parameters
    s : ndarray
        State vectors
        
    Returns:
    --------
    ndarray
        Energy contribution from pairwise interactions
    """
    e = np.zeros(s.shape[0])
    for n in range(s.shape[0]):
        k = 0
        for i in range(s.shape[1]-1):
            for j in range(i+1, s.shape[1]):
                e[n] += J[k] * s[n, i] * s[n, j]
                k += 1
    return e

@njit("float64[:](int64[:,:],float64[:])")
def calc_e(s, params):
    """
    Calculate energy for given states using the Ising model.
    
    Parameters:
    -----------
    s : 2D ndarray of ints
        State vectors, either {0,1} or {+/-1}
    params : ndarray
        (h, J) parameter vector containing local fields and couplings
        
    Returns:
    --------
    ndarray
        Energies of all given states
    """
    e = -fast_sum(params[s.shape[1]:], s)
    e -= np.sum(s * params[:s.shape[1]], 1)
    return e

def metropolis(initial_v, multiplier, temp, bootStrap=100000, samples=100000):
    """
    Metropolis algorithm for sampling from the Ising model distribution.
    Includes temperature as a parameter to study phase transitions.
    
    Parameters:
    -----------
    initial_v : ndarray
        Initial state vector
    multiplier : ndarray
        Model parameters (h, J)
    temp : float
        Temperature parameter
    bootStrap : int, optional
        Number of samples to discard at the beginning (default=1000)
    samples : int, optional
        Total number of samples to generate (default=100000)
        
    Returns:
    --------
    tuple
        (final_vector, net_spin_history, energy_history)
    """
    net_spin = []
    energy_spin = []
    
    current_vec = initial_v.copy()
    for i in range(0, samples):
        E_i = calc_e(current_vec.reshape(1, -1), multiplier)[0]

        # Permutate vector
        index = np.random.randint(0, high=current_vec.shape[1])
        mu_vector = current_vec.copy()
        mu_vector[:, index] *= -1

        E_u = calc_e(mu_vector.reshape(1, -1), multiplier)[0]

        # Accept or reject altered vector
        dE = E_u - E_i
        if (dE > 0) * (np.random.random() < np.exp(-temp * dE)):
            current_vec = mu_vector
        elif dE <= 0:
            current_vec = mu_vector

        # Store data after burn-in period
        if i > samples - bootStrap:
            net_spin += [current_vec.sum()]
            energy_spin += [calc_e(current_vec.reshape(1, -1), multiplier)[0]]

    return current_vec, net_spin, energy_spin

def heat_capacity(energy_samples):
    """
    Calculate heat capacity from energy samples.
    
    Parameters:
    -----------
    energy_samples : list or ndarray
        List of energy values
        
    Returns:
    --------
    float
        Heat capacity value
    """
    return np.power(np.asarray(energy_samples), 2).mean() - np.power(np.asarray(energy_samples).mean(), 2)

def fit_ising_model(spike_data, sample_size=10000, n_cpus=8, max_iter=75, eta=1e-3, rng_seed=0):
    """
    Fit an Ising model to the spike data using Maximum Entropy principles.
    
    Parameters:
    -----------
    spike_data : ndarray
        Preprocessed spike data
    sample_size : int, optional
        Sample size for the solver (default=10000)
    n_cpus : int, optional
        Number of CPUs to use (default=8)
    max_iter : int, optional
        Maximum number of iterations (default=75)
    eta : float, optional
        Learning rate (default=1e-3)
    rng_seed : int, optional
        Random number generator seed (default=0)
        
    Returns:
    --------
    ndarray
        Model parameters (h, J)
    """
    # Convert spikes to {-1, 1} representation if they're in {0, 1}
    if np.min(spike_data) == 0:
        spike_data = 2 * spike_data - 1
    
    N = spike_data.shape[1]
    
    # Initialize the solver
    solver = MCH(spike_data,
                 sample_size=sample_size,
                 rng=np.random.RandomState(rng_seed),
                 n_cpus=n_cpus,
                 sampler_kw={'boost': True})
                 
    # Define learning settings
    def learn_settings(i):
        print(f"Iteration {i}")
        return {'maxdlamda': 1, 'eta': eta}
    
    # Solve for model parameters
    start_time = time.time()
    multipliers = solver.solve(maxiter=max_iter,
                              n_iters=N*10,
                              burn_in=N*10,
                              iprint="detailed",
                              custom_convergence_f=learn_settings)
    
    print(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
    
    return multipliers, solver

def phase_transition_analysis(multipliers, N, temp_range=None, samples=1000000):
    """
    Analyze phase transitions in the Ising model by varying temperature.
    
    Parameters:
    -----------
    multipliers : ndarray
        Model parameters (h, J)
    N : int
        Number of neurons
    temp_range : ndarray, optional
        Range of temperatures to scan (default is np.arange(0.1, 2, 0.05))
    samples : int, optional
        Number of samples per temperature (default=1000000)
        
    Returns:
    --------
    dict
        Dictionary containing various thermodynamic quantities
    """
    if temp_range is None:
        temp_range = np.arange(0.1, 2, 0.05)
    
    # Initialize with opposite starting configurations
    vec_neg = 2 * (np.random.random(size=(1, N)) > 0.90) - 1
    vec_pos = vec_neg * -1

    pos_avg_spin = []
    neg_avg_spin = []

    pos_avg_energy_c = []
    neg_avg_energy_c = []

    for temp in temp_range:
        print(f"Temperature: {temp}")
        
        # Run Metropolis sampling with both initializations
        _, net_spin_neg, net_energy_neg = metropolis(vec_neg.copy(), multipliers, 1/temp, samples=samples)
        _, net_spin_pos, net_energy_pos = metropolis(vec_pos.copy(), multipliers, 1/temp, samples=samples)

        pos_avg_spin += [net_spin_pos]
        neg_avg_spin += [net_spin_neg]

        pos_avg_energy_c += [net_energy_pos]
        neg_avg_energy_c += [net_energy_neg]
    
    # Calculate derived quantities
    pos_avg_pos = [np.mean(r)/N for r in pos_avg_spin]
    pos_avg_energy = [np.mean(r) for r in pos_avg_energy_c]
    neg_avg_pos = [np.mean(r)/N for r in neg_avg_spin]
    neg_avg_energy = [np.mean(r) for r in neg_avg_energy_c]

    avg_heat_capacity_pos = [heat_capacity(r) for r in pos_avg_energy_c]
    avg_heat_capacity_neg = [heat_capacity(r) for r in neg_avg_energy_c]
    
    # Find critical temperature
    critical_temp_idx = np.argmax(avg_heat_capacity_pos)
    c_temp = temp_range[critical_temp_idx]
    critical_energy = (pos_avg_energy[critical_temp_idx] + neg_avg_energy[critical_temp_idx])/2
    
    # Return all results
    return {
        'temp_range': temp_range,
        'pos_avg_pos': pos_avg_pos,
        'neg_avg_pos': neg_avg_pos,
        'pos_avg_energy': pos_avg_energy,
        'neg_avg_energy': neg_avg_energy,
        'avg_heat_capacity_pos': avg_heat_capacity_pos,
        'avg_heat_capacity_neg': avg_heat_capacity_neg,
        'critical_temp': c_temp,
        'critical_energy': critical_energy,
        'critical_temp_idx': critical_temp_idx
    }

def calculate_energy_for_spike_data(neural_data, model_params):
    """
    Calculate energy for each spike configuration in the data.
    
    Parameters:
    -----------
    neural_data : list of ndarrays
        List of spike data arrays
    model_params : ndarray
        Model parameters (h, J)
        
    Returns:
    --------
    list of ndarrays
        List of energy values for each spike configuration
    """
    # Convert to binary if needed
    if not isinstance(neural_data[0], np.ndarray):
        neural_data = np.asarray(neural_data)
    
    # Calculate energy for each session
    energies = []
    for session_data in neural_data:
        # Ensure binary encoding
        binary_data = (session_data > 0) * 1
        
        # Calculate energy
        session_energy = np.array([calc_e(binary_data, model_params)])
        energies.append(session_energy)
    
    return energies