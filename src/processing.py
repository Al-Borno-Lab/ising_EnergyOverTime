from multiprocessing import Pool
import numpy as np

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
def calculate_session_averages(df: pd.DataFrame, confidence: float = 0.8) -> Dict:
    """
    Calculate average statistics across all reaches in a session for kinematic 
    and neural data from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: reach_idx, stim, x, y, z, firing_rate, energy, j, h
        Each row represents a single time point for a specific reach in a specific stimulus
    confidence : float, optional
        Confidence level for intervals (default=0.8)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'by_stimulus': Dict with stats for each stimulus condition
            - Each stimulus contains: x, y, z, firing_rate, energy, j, h
            - Each variable has: mean (over time), lower_ci, upper_ci, time_series_mean, 
              time_series_lower, time_series_upper
        - 'overall': Dict with overall session averages across all stimuli
        - 'metadata': Dict with session information (n_reaches, n_stimuli, n_timepoints)
    """
    
    def mean_confidence_interval(data: np.ndarray, confidence: float = 0.8):
        """Calculate mean and confidence interval for data."""
        import scipy.stats as stats
        
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
        if n == 1:
            return data[0], data[0], data[0]
        
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return mean, mean - h, mean + h
    
    # Get unique stimuli
    stimuli = sorted(df['stim'].unique())
    
    # First, calculate 3D velocity for each reach
    df = df.copy()  # Don't modify original
    velocity_data = []
    
    for (reach_idx, stim), group in df.groupby(['reach_idx', 'stim']):
        group = group.sort_values('time_idx') if 'time_idx' in group.columns else group
        
        x_vals = group['x'].values
        
        # Calculate x velocity as position change (assuming dt=1)
        dx = np.diff(x_vals, prepend=x_vals[0])
        velocity_x = dx
        
        # Calculate x acceleration as velocity change (assuming dt=1)
        acceleration_x = np.diff(velocity_x, prepend=velocity_x[0])
        
        for i, (vel, acc) in enumerate(zip(velocity_x, acceleration_x)):
            velocity_data.append({
                'reach_idx': reach_idx,
                'stim': stim,
                'velocity_x': vel,
                'acceleration_x': acc
            })
    
    # Merge velocity and acceleration back into dataframe
    velocity_df = pd.DataFrame(velocity_data)
    df = df.reset_index(drop=True)
    df['velocity_x'] = velocity_df['velocity_x'].values
    df['acceleration_x'] = velocity_df['acceleration_x'].values
    
    # Variables to process (now including velocity_3d)
    variables = ['x', 'y', 'z', 'velocity_x', 'acceleration_x', 'firing_rate', 'energy', 'j', 'h']
    
    # Initialize results
    results = {
        'by_stimulus': {},
        'overall': {},
        'metadata': {
            'n_stimuli': len(stimuli),
            'n_reaches_total': df['reach_idx'].nunique(),
            'confidence_level': confidence
        }
    }
    
    # Collect all data for overall averages
    all_data = {var: [] for var in variables}
    
    # Calculate statistics for each stimulus condition
    for stim in stimuli:
        stim_data = df[df['stim'] == stim].copy()
        
        # Add time index within each reach
        stim_data['time_idx'] = stim_data.groupby('reach_idx').cumcount()
        max_time = stim_data['time_idx'].max() + 1
        
        n_reaches = stim_data['reach_idx'].nunique()
        
        # Initialize storage for this stimulus
        stim_results = {
            'n_reaches': n_reaches,
            'n_timepoints': max_time
        }
        
        for var in variables:
            if var not in stim_data.columns:
                continue
                
            # Time series statistics (mean across reaches at each time point)
            time_series = {'mean': [], 'lower': [], 'upper': []}
            
            for t in range(max_time):
                time_slice = stim_data[stim_data['time_idx'] == t]
                if len(time_slice) > 0:
                    values = time_slice[var].values
                    m, l, u = mean_confidence_interval(values, confidence)
                    time_series['mean'].append(m)
                    time_series['lower'].append(l)
                    time_series['upper'].append(u)
                    
                    # Collect for overall averages
                    all_data[var].extend(values)
            
            # Convert to numpy arrays
            time_series['mean'] = np.array(time_series['mean'])
            time_series['lower'] = np.array(time_series['lower'])
            time_series['upper'] = np.array(time_series['upper'])
            
            # Calculate scalar summary statistics
            all_values = stim_data[var].values
            overall_mean, overall_lower, overall_upper = mean_confidence_interval(all_values, confidence)
            
            stim_results[var] = {
                # Scalar summaries
                'mean': overall_mean,
                'lower_ci': overall_lower,
                'upper_ci': overall_upper,
                'std': np.std(all_values),
                'min': np.min(all_values),
                'max': np.max(all_values),
                # Time series
                'time_series_mean': time_series['mean'],
                'time_series_lower': time_series['lower'],
                'time_series_upper': time_series['upper']
            }
        
        results['by_stimulus'][stim] = stim_results
    
    # Calculate overall session averages (across all stimuli)
    for var in variables:
        if all_data[var]:
            values = np.array(all_data[var])
            overall_mean, overall_lower, overall_upper = mean_confidence_interval(values, confidence)
            
            results['overall'][var] = {
                'mean': overall_mean,
                'lower_ci': overall_lower,
                'upper_ci': overall_upper,
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return results


def calculate_reach_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average value of each variable for each reach, including 3D velocity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: reach_idx, stim, x, y, z, firing_rate, energy, j, h
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per reach, containing average values for each variable
        including velocity_3d
    """
    df = df.copy()
    
    # Calculate 3D velocity for each reach
    reach_velocities = []
    
    for (reach_idx, stim), group in df.groupby(['reach_idx', 'stim']):
        x_vals = group['x'].values
        y_vals = group['y'].values
        z_vals = group['z'].values
        
        # Calculate velocity as magnitude of position change
        dx = np.diff(x_vals, prepend=x_vals[0])
        dy = np.diff(y_vals, prepend=y_vals[0])
        dz = np.diff(z_vals, prepend=z_vals[0])
        
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        avg_velocity = np.mean(velocity)
        
        reach_velocities.append({
            'reach_idx': reach_idx,
            'stim': stim,
            'velocity_3d': avg_velocity
        })
    
    variables = ['x', 'y', 'z', 'firing_rate', 'energy', 'j', 'h']
    available_vars = [v for v in variables if v in df.columns]
    
    # Group by reach and stimulus, calculate mean of each variable
    reach_averages = df.groupby(['reach_idx', 'stim'])[available_vars].mean().reset_index()
    
    # Merge velocity data
    velocity_df = pd.DataFrame(reach_velocities)
    reach_averages = reach_averages.merge(velocity_df, on=['reach_idx', 'stim'])
    
    return reach_averages


def get_session_summary(df: pd.DataFrame, confidence: float = 0.8) -> pd.DataFrame:
    """
    TODO: INCLUDE ACCELERATION
    Get a summary DataFrame of session averages by stimulus.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: reach_idx, stim, x, y, z, firing_rate, energy, j, h
    confidence : float, optional
        Confidence level for intervals (default=0.8)
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with stimulus as index and variable statistics as columns
    """
    stats = calculate_session_averages(df, confidence)
    
    # Build summary table
    summary_rows = []
    
    for stim, stim_data in stats['by_stimulus'].items():
        row = {'stimulus': stim, 'n_reaches': stim_data['n_reaches']}
        
        for var in ['x', 'y', 'z', 'velocity_3d', 'firing_rate', 'energy', 'j', 'h']:
            if var in stim_data:
                row[f'{var}_mean'] = stim_data[var]['mean']
                row[f'{var}_std'] = stim_data[var]['std']
                row[f'{var}_lower_ci'] = stim_data[var]['lower_ci']
                row[f'{var}_upper_ci'] = stim_data[var]['upper_ci']
        
        summary_rows.append(row)
    
    # Add overall row
    overall_row = {'stimulus': 'OVERALL', 'n_reaches': stats['metadata']['n_reaches_total']}
    for var in ['x', 'y', 'z', 'velocity_3d', 'firing_rate', 'energy', 'j', 'h']:
        if var in stats['overall']:
            overall_row[f'{var}_mean'] = stats['overall'][var]['mean']
            overall_row[f'{var}_std'] = stats['overall'][var]['std']
            overall_row[f'{var}_lower_ci'] = stats['overall'][var]['lower_ci']
            overall_row[f'{var}_upper_ci'] = stats['overall'][var]['upper_ci']
    
    summary_rows.append(overall_row)
    
    return pd.DataFrame(summary_rows)


# Helper function to find extrema within a constrained range
def find_extrema_in_range(data, start_idx, end_idx):
    """Find min and max indices within the specified range."""
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(data), end_idx)
    
    if start_idx >= end_idx or start_idx >= len(data):
        # Fallback to global if range is invalid
        return np.argmin(data), np.argmax(data)
    
    # Get the slice and find extrema
    data_slice = data[start_idx:end_idx]
    local_min_idx = np.argmin(data_slice)
    local_max_idx = np.argmax(data_slice)
    
    # Convert back to global indices
    global_min_idx = start_idx + local_min_idx
    global_max_idx = start_idx + local_max_idx
    
    return global_min_idx, global_max_idx

def process_session(args):
    stim, session, data, window = args
    
    stats = calculate_session_averages(data, confidence=0.8)
    
    data_frame = {
        'acceleration':  find_extrema_in_range(stats['by_stimulus'][stim]['acceleration_x']['time_series_mean'], window[0], window[1]),
        'velocity':      find_extrema_in_range(stats['by_stimulus'][stim]['velocity_x']['time_series_mean'], window[0], window[1]),
        'firing_rate':   find_extrema_in_range(stats['by_stimulus'][stim]['firing_rate']['time_series_mean'],  window[0], window[1]),
        'energy':        find_extrema_in_range(stats['by_stimulus'][stim]['energy']['time_series_mean'],        window[0], window[1]),
        'Boltzman_prob': find_extrema_in_range(np.exp(-stats['by_stimulus'][stim]['energy']['time_series_mean']), window[0], window[1]),
        'original_data': data
    }
    
    print(f"✓ Done: stim={stim}, session={session}")
    return stim, session, data_frame
    