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
from coniii import *
from multiprocessing import Pool

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


@njit("Tuple((float64[:], float64[:], float64[:]))(int64[:,:],float64[:])")
def calc_e_with_terms(s, params):
    """
    Calculate energy for given states with individual term contributions.
    
    Parameters:
    -----------
    s : 2D ndarray of ints
        State vectors, either {0,1} or {+/-1}
    params : ndarray
        (h, J) parameter vector containing local fields and couplings
        
    Returns:
    --------
    tuple
        (energies, J_contributions, h_contributions)
    """
    j = fast_sum(params[s.shape[1]:], s) # Pairwise interaction contributions
    h = np.sum(s * params[:s.shape[1]], 1) # Local field contributions
    e = -j - h
    return (e, j, h)


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
        # if (dE > 0) * (np.random.random() < np.exp(-temp * dE)):
        #     current_vec = mu_vector
        # elif dE <= 0:
        #     current_vec = mu_vector

        if dE <= 0:
            current_vec = mu_vector
        elif np.random.random() < np.exp(-dE / temp):
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


# def fit_ising_model(spike_data, sample_size=10000, n_cpus=8, max_iter=75, eta=1e-3, rng_seed=0):
#     """
#     Fit an Ising model to the spike data using Maximum Entropy principles.
    
#     Parameters:
#     -----------
#     spike_data : ndarray
#         Preprocessed spike data
#     sample_size : int, optional
#         Sample size for the solver (default=10000)
#     n_cpus : int, optional
#         Number of CPUs to use (default=8)
#     max_iter : int, optional
#         Maximum number of iterations (default=75)
#     eta : float, optional
#         Learning rate (default=1e-3)
#     rng_seed : int, optional
#         Random number generator seed (default=0)
        
#     Returns:
#     --------
#     ndarray
#         Model parameters (h, J)
#     """
#     # Convert spikes to {-1, 1} representation if they're in {0, 1}
#     if np.min(spike_data) == 0:
#         spike_data = 2 * spike_data - 1
    
#     N = spike_data.shape[1]
    
#     # Initialize the solver
#     # solver = Pseudo(spike_data)
#     # solver = MCH(spike_data,
#     #              sample_size=sample_size,
#     #              rng=np.random.RandomState(rng_seed),
#     #              n_cpus=n_cpus,
#     #              sampler_kw={'boost': True})
                 
#     # Define learning settings
#     # def learn_settings(i):
#     #     print(f"Iteration {i}")
#     #     return {'maxdlamda': 1, 'eta': eta}
    
#     # Solve for model parameters
#     start_time = time.time()
#     multipliers = solver.solve()
    
#     print(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
    
#     return multipliers, solver


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
    multipliers = solver.solve( maxiter=max_iter,
                                n_iters=max(N*100, 2000),
                                burn_in=max(N*50, 1000),
                                iprint="detailed",
                                custom_convergence_f=learn_settings)
    
    print(f"Model fitting completed in {time.time() - start_time:.2f} seconds")
    
    return multipliers, solver

def _process_single_temperature(args):
    """
    Helper function to process a single temperature for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (temp, multipliers, vec_pos, vec_neg, N, samples)
        
    Returns:
    --------
    tuple
        (temp, net_spin_pos, net_spin_neg, net_energy_pos, net_energy_neg)
    """
    temp, multipliers, vec_pos, vec_neg, N, samples = args
    print(f"Temperature: {temp}")
    
    # Run Metropolis sampling with both initializations
    _, net_spin_neg, net_energy_neg = metropolis(vec_neg.copy(), multipliers, temp, samples=samples)
    _, net_spin_pos, net_energy_pos = metropolis(vec_pos.copy(), multipliers, temp, samples=samples)
    
    return (temp, net_spin_pos, net_spin_neg, net_energy_pos, net_energy_neg)


def phase_transition_analysis(multipliers, N, temp_range=None, samples=1000000, num_cores=1):
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
    num_cores : int, optional
        Number of CPU cores to use for parallel processing (default=1)
        
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

    # Prepare arguments for multiprocessing
    process_args = [
        (temp, multipliers, vec_pos.copy(), vec_neg.copy(), N, samples)
        for temp in temp_range
    ]

    # Process temperatures in parallel or sequentially
    if num_cores > 1:
        print(f"Using {num_cores} cores for parallel processing...")
        with Pool(processes=num_cores) as pool:
            results = pool.map(_process_single_temperature, process_args)
    else:
        print("Using sequential processing...")
        results = [_process_single_temperature(args) for args in process_args]
    
    # Sort results by temperature to maintain order
    results.sort(key=lambda x: x[0])
    
    # Extract results
    pos_avg_spin = []
    neg_avg_spin = []
    pos_avg_energy_c = []
    neg_avg_energy_c = []
    
    for _, net_spin_pos, net_spin_neg, net_energy_pos, net_energy_neg in results:
        pos_avg_spin.append(net_spin_pos)
        neg_avg_spin.append(net_spin_neg)
        pos_avg_energy_c.append(net_energy_pos)
        neg_avg_energy_c.append(net_energy_neg)
    
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

def _process_single_k_value(args):
    """
    Helper function to process a single k value for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (k, model_params, N, n_samples, process_seed)
        
    Returns:
    --------
    tuple
        (k, energies_k)
    """
    k, model_params, N, n_samples, process_seed = args
    
    # Set random seed for this process
    if process_seed is not None:
        np.random.seed(process_seed)
    
    # Generate random configurations with exactly k neurons on
    energies_k = []
    
    for _ in range(n_samples):
        # Create vector with k neurons on (+1) and rest off (-1)
        vec = np.ones(N, dtype=np.int64) * -1  # Start with all off
        # Randomly select k indices to turn on
        on_indices = np.random.choice(N, size=k, replace=False)
        vec[on_indices] = 1
        
        # Reshape to 2D for calc_e (needs shape (1, N))
        vec_2d = vec.reshape(1, -1)
        
        # Calculate energy
        energy = calc_e(vec_2d, model_params)[0]
        energies_k.append(energy)
    
    print(f"Calculated {n_samples} energies for k={k} neurons on")
    return (k, energies_k)


def calculate_energy_by_neuron_count(model_params, N, num_samples_per_count=None, rng_seed=None, num_cores=1):
    """
    Calculate energy for random configurations with k neurons on (k from 1 to N-1).
    
    For each k (number of neurons on), randomly generates configurations with exactly
    k neurons set to +1 (on) and the rest set to -1 (off), then calculates the energy
    for each configuration.
    
    Parameters:
    -----------
    model_params : ndarray
        Model parameters (h, J)
    N : int
        Number of neurons
    num_samples_per_count : int, optional
        Number of random configurations to generate for each k (default: N choose k, 
        or 1000 if that's too large)
    rng_seed : int, optional
        Random number generator seed for reproducibility (default: None)
    num_cores : int, optional
        Number of CPU cores to use for parallel processing (default=1)
        
    Returns:
    --------
    dict
        Dictionary with keys 'k_values' (list of k values) and 'energies_by_k' 
        (list of lists, where energies_by_k[i] contains energies for k=k_values[i])
    """
    k_values = list(range(1, N))
    
    # Determine number of samples for each k
    if num_samples_per_count is None:
        # Calculate number of possible combinations for each k
        n_samples_list = []
        for k in k_values:
            try:
                from math import comb
                max_combinations = comb(N, k)
            except ImportError:
                try:
                    from scipy.special import comb
                    max_combinations = int(comb(N, k, exact=True))
                except (ImportError, ValueError):
                    max_combinations = 10000
            n_samples_list.append(min(max_combinations, 1000))
    else:
        n_samples_list = [num_samples_per_count] * len(k_values)
    
    # Prepare arguments for multiprocessing
    # Create unique seeds for each process if rng_seed is provided
    if rng_seed is not None:
        process_seeds = [rng_seed + i for i in range(len(k_values))]
    else:
        process_seeds = [None] * len(k_values)
    
    process_args = [
        (k, model_params, N, n_samples, seed)
        for k, n_samples, seed in zip(k_values, n_samples_list, process_seeds)
    ]
    
    # Process k values in parallel or sequentially
    if num_cores > 1:
        print(f"Using {num_cores} cores for parallel processing...")
        with Pool(processes=num_cores) as pool:
            results = pool.map(_process_single_k_value, process_args)
    else:
        print("Using sequential processing...")
        results = [_process_single_k_value(args) for args in process_args]
    
    # Sort results by k to maintain order
    results.sort(key=lambda x: x[0])
    
    # Extract energies in order
    energies_by_k = [energies for _, energies in results]
    
    return {
        'k_values': k_values,
        'energies_by_k': energies_by_k
    }

