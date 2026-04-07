import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict


def within_session_test(stim_sessions_extrema: Dict, 
                        verbose: bool = True) -> Dict:
    """
    Within-session test: Compare SAME-SESSION energy vs SAME-SESSION firing rate.
    
    This test answers: "Within each session, does energy or firing rate 
    have extrema closer to max velocity?"
    
    Also calculates covariance between energy and firing rate time series.
    
    No cross-session comparisons - purely within-session analysis.
    
    Parameters:
    -----------
    stim_sessions_extrema : Dict
        Dictionary structured as:
        {
            stim_idx: {
                session_id: {
                    'acceleration': (min_idx, max_idx), 
                    'velocity': (min_idx, max_idx),
                    'energy': (min_idx, max_idx),
                    'firing_rate': (min_idx, max_idx),
                    'original_data': pd.DataFrame (optional, for covariance calculation)
                }
            }
        }
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    Dict containing within-session test results
    """
    
    results = {
        'by_stimulus': {},
        'overall': {},
        'all_sessions': []
    }
    
    # Overall counters
    total_energy_wins = 0
    total_firing_wins = 0
    total_ties = 0
    total_sessions = 0
    
    all_energy_distances = []
    all_firing_distances = []
    all_covariances = []
    all_correlations = []
    
    # Track which extremum type wins
    extremum_wins = {
        'energy_min': 0,
        'energy_max': 0,
        'firing_min': 0,
        'firing_max': 0
    }
    
    if verbose:
        print("=" * 80)
        print("WITHIN-SESSION TEST")
        print("Comparing SAME-SESSION energy vs SAME-SESSION firing rate")
        print("=" * 80)
        print("\nQuestion: Within each session, which measure's extrema")
        print("          is closest to that session's max velocity?")
        print("=" * 80)
    
    for stim in sorted(stim_sessions_extrema.keys()):
        sessions = stim_sessions_extrema[stim]
        session_list = list(sessions.keys())
        n_sessions = len(session_list)
        
        # Counters for this stimulus
        energy_wins = 0
        firing_wins = 0
        ties = 0
        stim_sessions_list = []
        
        energy_distances = []
        firing_distances = []
        stim_covariances = []
        stim_correlations = []
        
        stim_extremum_wins = {
            'energy_min': 0,
            'energy_max': 0,
            'firing_min': 0,
            'firing_max': 0
        }
        
        for session in session_list:
            max_vel_idx = sessions[session]['velocity'][1]
            max_accel_idx = sessions[session]['acceleration'][1]
            
            # Calculate covariance if original_data is available
            covariance = np.nan
            correlation = np.nan
            if 'original_data' in sessions[session]:
                original_data = sessions[session]['original_data']
                original_data_stim = original_data[original_data['stim'] == stim]
                
                if len(original_data_stim) > 0:
                    energy_ts = _calculate_mean_timeseries(original_data_stim, 'energy')
                    firing_ts = _calculate_mean_timeseries(original_data_stim, 'firing_rate')
                    
                    if len(energy_ts) > 0 and len(firing_ts) > 0:
                        min_len = min(len(energy_ts), len(firing_ts))
                        energy_ts = energy_ts[:min_len]
                        firing_ts = firing_ts[:min_len]
                        
                        covariance = np.cov(energy_ts, firing_ts)[0, 1]
                        correlation = np.corrcoef(energy_ts, firing_ts)[0, 1]
                        
                        stim_covariances.append(covariance)
                        stim_correlations.append(correlation)
                        all_covariances.append(covariance)
                        all_correlations.append(correlation)
            
            # Get energy distances
            energy_min_idx = sessions[session]['energy'][0]
            energy_max_idx = sessions[session]['energy'][1]
            energy_min_dist = abs(max_accel_idx - energy_min_idx)
            energy_max_dist = abs(max_accel_idx - energy_max_idx)
            
            if energy_min_dist <= energy_max_dist:
                energy_best_dist = energy_min_dist
                energy_best_type = 'min'
                energy_best_idx = energy_min_idx
            else:
                energy_best_dist = energy_max_dist
                energy_best_type = 'max'
                energy_best_idx = energy_max_idx
            
            # Get firing rate distances
            firing_min_idx = sessions[session]['firing_rate'][0]
            firing_max_idx = sessions[session]['firing_rate'][1]
            firing_min_dist = abs(max_accel_idx - firing_min_idx)
            firing_max_dist = abs(max_accel_idx - firing_max_idx)
            
            if firing_min_dist <= firing_max_dist:
                firing_best_dist = firing_min_dist
                firing_best_type = 'min'
                firing_best_idx = firing_min_idx
            else:
                firing_best_dist = firing_max_dist
                firing_best_type = 'max'
                firing_best_idx = firing_max_idx
            
            # Record distances
            energy_distances.append(energy_best_dist)
            firing_distances.append(firing_best_dist)
            all_energy_distances.append(energy_best_dist)
            all_firing_distances.append(firing_best_dist)
            
            # Determine winner
            if energy_best_dist < firing_best_dist:
                energy_wins += 1
                winner = 'energy'
                if energy_best_type == 'min':
                    stim_extremum_wins['energy_min'] += 1
                    extremum_wins['energy_min'] += 1
                else:
                    stim_extremum_wins['energy_max'] += 1
                    extremum_wins['energy_max'] += 1
            elif firing_best_dist < energy_best_dist:
                firing_wins += 1
                winner = 'firing'
                if firing_best_type == 'min':
                    stim_extremum_wins['firing_min'] += 1
                    extremum_wins['firing_min'] += 1
                else:
                    stim_extremum_wins['firing_max'] += 1
                    extremum_wins['firing_max'] += 1
            else:
                ties += 1
                winner = 'tie'
            
            session_result = {
                'stimulus': stim,
                'session': session,
                'max_vel_idx': max_vel_idx,
                'max_accel_idx': max_accel_idx,
                'energy_min_idx': energy_min_idx,
                'energy_max_idx': energy_max_idx,
                'energy_best_dist': energy_best_dist,
                'energy_best_type': energy_best_type,
                'energy_best_idx': energy_best_idx,
                'firing_min_idx': firing_min_idx,
                'firing_max_idx': firing_max_idx,
                'firing_best_dist': firing_best_dist,
                'firing_best_type': firing_best_type,
                'firing_best_idx': firing_best_idx,
                'winner': winner,
                'covariance': covariance,
                'correlation': correlation
            }
            stim_sessions_list.append(session_result)
            results['all_sessions'].append(session_result)
        
        total_energy_wins += energy_wins
        total_firing_wins += firing_wins
        total_ties += ties
        total_sessions += n_sessions
        
        results['by_stimulus'][stim] = {
            'energy_wins': energy_wins,
            'firing_wins': firing_wins,
            'ties': ties,
            'n_sessions': n_sessions,
            'energy_win_pct': 100 * energy_wins / n_sessions if n_sessions > 0 else 0,
            'mean_energy_distance': np.mean(energy_distances),
            'mean_firing_distance': np.mean(firing_distances),
            'std_energy_distance': np.std(energy_distances),
            'std_firing_distance': np.std(firing_distances),
            'mean_covariance': np.nanmean(stim_covariances) if stim_covariances else np.nan,
            'mean_correlation': np.nanmean(stim_correlations) if stim_correlations else np.nan,
            'extremum_wins': stim_extremum_wins,
            'sessions': stim_sessions_list
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"STIMULUS {stim} ({n_sessions} sessions)")
            print("=" * 80)
            print(f"\n--- Win Counts ---")
            print(f"  ENERGY wins:      {energy_wins} ({100*energy_wins/n_sessions:.1f}%)")
            print(f"  FIRING RATE wins: {firing_wins} ({100*firing_wins/n_sessions:.1f}%)")
            print(f"  Ties:             {ties} ({100*ties/n_sessions:.1f}%)")
            
            print(f"\n--- Distance Statistics ---")
            print(f"  Energy:      mean = {np.mean(energy_distances):.2f} ± {np.std(energy_distances):.2f}")
            print(f"  Firing rate: mean = {np.mean(firing_distances):.2f} ± {np.std(firing_distances):.2f}")
            
            if stim_covariances:
                print(f"\n--- Covariance/Correlation Statistics ---")
                print(f"  Mean Covariance:  {np.nanmean(stim_covariances):.4f}")
                print(f"  Mean Correlation: {np.nanmean(stim_correlations):.4f}")
            
            print(f"\n--- Extremum Breakdown ---")
            print(f"  energy_min wins:  {stim_extremum_wins['energy_min']}")
            print(f"  energy_max wins:  {stim_extremum_wins['energy_max']}")
            print(f"  firing_min wins:  {stim_extremum_wins['firing_min']}")
            print(f"  firing_max wins:  {stim_extremum_wins['firing_max']}")
    
    # Overall results
    results['overall'] = {
        'energy_wins': total_energy_wins,
        'firing_wins': total_firing_wins,
        'ties': total_ties,
        'n_sessions': total_sessions,
        'energy_win_pct': 100 * total_energy_wins / total_sessions if total_sessions > 0 else 0,
        'mean_energy_distance': np.mean(all_energy_distances),
        'mean_firing_distance': np.mean(all_firing_distances),
        'std_energy_distance': np.std(all_energy_distances),
        'std_firing_distance': np.std(all_firing_distances),
        'mean_covariance': np.nanmean(all_covariances) if all_covariances else np.nan,
        'mean_correlation': np.nanmean(all_correlations) if all_correlations else np.nan,
        'extremum_wins': extremum_wins
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("OVERALL RESULTS (ALL STIMULI COMBINED)")
        print("=" * 80)
        
        print(f"\n--- Overall Totals ---")
        print(f"Total sessions: {total_sessions}")
        print(f"\n  ENERGY wins:      {total_energy_wins} ({100*total_energy_wins/total_sessions:.1f}%)")
        print(f"  FIRING RATE wins: {total_firing_wins} ({100*total_firing_wins/total_sessions:.1f}%)")
        print(f"  Ties:             {total_ties} ({100*total_ties/total_sessions:.1f}%)")
        
        if all_covariances:
            print(f"\n--- Overall Covariance/Correlation ---")
            print(f"  Mean Covariance:  {np.nanmean(all_covariances):.4f}")
            print(f"  Mean Correlation: {np.nanmean(all_correlations):.4f}")
    
    return results


def within_session_test_with_plots(stim_sessions_extrema: Dict, 
                                    output_dir: str,
                                    verbose: bool = True) -> Dict:
    """
    Within-session test with visualization and covariance calculation.
    
    Generates plots showing:
    - Top: Session's velocity time series with max velocity marked
    - Middle: Session's energy time series with extrema marked
    - Bottom: Session's firing rate time series with extrema marked
    - Title includes covariance and correlation between energy and firing rate
    
    Plots are organized into folders based on winner (energy_wins, firing_wins, ties).
    Also outputs a summary CSV with all sessions, winners, and covariances.
    
    Parameters:
    -----------
    stim_sessions_extrema : Dict
        Dictionary with 'original_data' containing the dataframe
    output_dir : str
        Base output directory for saving plots
    verbose : bool
        Whether to print detailed results
        
    Returns:
    --------
    Dict containing within-session test results
    """
    
    # Create output directories
    energy_wins_dir = os.path.join(output_dir, 'energy_wins')
    firing_wins_dir = os.path.join(output_dir, 'firing_wins')
    ties_dir = os.path.join(output_dir, 'ties')
    
    os.makedirs(energy_wins_dir, exist_ok=True)
    os.makedirs(firing_wins_dir, exist_ok=True)
    os.makedirs(ties_dir, exist_ok=True)
    
    # First run the basic test to get results
    results = within_session_test(stim_sessions_extrema, verbose=verbose)
    
    if verbose:
        print(f"\n--- Generating Plots ---")
    
    # Generate plots for each session
    for session_data in results['all_sessions']:
        stim = session_data['stimulus']
        session = session_data['session']
        
        # Get original data
        if 'original_data' not in stim_sessions_extrema[stim][session]:
            if verbose:
                print(f"  Skipping {session} - no original_data")
            continue
        
        original_data = stim_sessions_extrema[stim][session]['original_data']
        original_data = original_data[original_data['stim'] == stim]
        
        # Calculate time series
        velocity_ts = _calculate_velocity_timeseries(original_data)
        acceleration_ts = _calculate_acceleration_timeseries(original_data)
        energy_ts = _calculate_mean_timeseries(original_data, 'energy')
        firing_ts = _calculate_mean_timeseries(original_data, 'firing_rate')
        
        # Determine save directory
        if session_data['winner'] == 'energy':
            save_dir = energy_wins_dir
        elif session_data['winner'] == 'firing':
            save_dir = firing_wins_dir
        else:
            save_dir = ties_dir

        
        # Create plot
        _create_within_session_plot(
            session=session,
            stim=stim,
            velocity_ts=velocity_ts,
            acceleration_ts=acceleration_ts, 
            max_accel_idx=session_data['max_accel_idx'],
            max_vel_idx=session_data['max_vel_idx'],
            energy_ts=energy_ts,
            energy_min_idx=session_data['energy_min_idx'],
            energy_max_idx=session_data['energy_max_idx'],
            energy_best_idx=session_data['energy_best_idx'],
            energy_best_dist=session_data['energy_best_dist'],
            energy_best_type=session_data['energy_best_type'],
            firing_ts=firing_ts,
            firing_min_idx=session_data['firing_min_idx'],
            firing_max_idx=session_data['firing_max_idx'],
            firing_best_idx=session_data['firing_best_idx'],
            firing_best_dist=session_data['firing_best_dist'],
            firing_best_type=session_data['firing_best_type'],
            winner=session_data['winner'],
            covariance=session_data['covariance'],
            correlation=session_data['correlation'],
            save_dir=save_dir
        )
    
    # Create summary CSV
    summary_df = create_summary_dataframe(results)
    summary_csv_path = os.path.join(output_dir, 'session_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    if verbose:
        print(f"\n--- Plots Saved ---")
        print(f"  Energy wins: {results['overall']['energy_wins']} plots in {energy_wins_dir}")
        print(f"  Firing wins: {results['overall']['firing_wins']} plots in {firing_wins_dir}")
        print(f"  Ties:        {results['overall']['ties']} plots in {ties_dir}")
        print(f"\n--- Summary CSV Saved ---")
        print(f"  {summary_csv_path}")
        print(f"\n--- Summary Table ---")
        print(summary_df.to_string(index=False))
    
    results['summary_df'] = summary_df
    
    return results


def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from within-session test results.
    
    Parameters:
    -----------
    results : Dict
        Results from within_session_test
        
    Returns:
    --------
    pd.DataFrame
        Summary table with all sessions, winners, distances, and covariances
    """
    rows = []
    
    for session_data in results['all_sessions']:
        rows.append({
            'Stimulus': session_data['stimulus'],
            'Session': session_data['session'],
            'Max_Velocity_Idx': session_data['max_vel_idx'],
            'Winner': session_data['winner'],
            'Energy_Best_Idx': session_data['energy_best_idx'],
            'Energy_Best_Type': session_data['energy_best_type'],
            'Energy_Distance': session_data['energy_best_dist'],
            'FiringRate_Best_Idx': session_data['firing_best_idx'],
            'FiringRate_Best_Type': session_data['firing_best_type'],
            'FiringRate_Distance': session_data['firing_best_dist'],
            'Covariance': session_data['covariance'],
            'Correlation': session_data['correlation']
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by stimulus then session
    df = df.sort_values(['Stimulus', 'Session']).reset_index(drop=True)
    
    return df

def _calculate_velocity_timeseries(df):
    """Calculate mean velocity time series from dataframe (x-component only)."""
    df = df.copy()
    df['time_idx'] = df.groupby('reach_idx').cumcount()
    
    velocity_data = []
    for reach_idx in df['reach_idx'].unique():
        reach_data = df[df['reach_idx'] == reach_idx].sort_values('time_idx')
        
        x = reach_data['x'].values
        
        dx = np.diff(x, prepend=x[0])
        
        for t, v in enumerate(dx):
            velocity_data.append({'time_idx': t, 'velocity': v})
    
    velocity_df = pd.DataFrame(velocity_data)
    mean_velocity = velocity_df.groupby('time_idx')['velocity'].mean().values
    
    return mean_velocity

def _calculate_acceleration_timeseries(df):
    """Calculate mean acceleration time series from dataframe (x-component only)."""
    df = df.copy()
    df['time_idx'] = df.groupby('reach_idx').cumcount()
    
    acceleration_data = []
    for reach_idx in df['reach_idx'].unique():
        reach_data = df[df['reach_idx'] == reach_idx].sort_values('time_idx')
        
        x = reach_data['x'].values
        
        dx = np.diff(x, prepend=x[0])
        ddx = np.diff(dx, prepend=dx[0])
        
        for t, a in enumerate(ddx):
            acceleration_data.append({'time_idx': t, 'acceleration': a})
    
    acceleration_df = pd.DataFrame(acceleration_data)
    mean_acceleration = acceleration_df.groupby('time_idx')['acceleration'].mean().values
    
    return mean_acceleration

# def _calculate_velocity_timeseries(df):
#     """Calculate mean velocity time series from dataframe."""
#     df = df.copy()
#     df['time_idx'] = df.groupby('reach_idx').cumcount()
    
#     velocity_data = []
#     for reach_idx in df['reach_idx'].unique():
#         reach_data = df[df['reach_idx'] == reach_idx].sort_values('time_idx')
        
#         x = reach_data['x'].values
#         y = reach_data['y'].values
#         z = reach_data['z'].values
        
#         dx = np.diff(x, prepend=x[0])
#         dy = np.diff(y, prepend=y[0])
#         dz = np.diff(z, prepend=z[0])
        
#         velocity = np.sqrt(dx**2 + dy**2 + dz**2)
        
#         for t, v in enumerate(velocity):
#             velocity_data.append({'time_idx': t, 'velocity': v})
    
#     velocity_df = pd.DataFrame(velocity_data)
#     mean_velocity = velocity_df.groupby('time_idx')['velocity'].mean().values
    
#     return mean_velocity


def _calculate_mean_timeseries(df, column):
    """Calculate mean time series for a given column from dataframe."""
    df = df.copy()
    df['time_idx'] = df.groupby('reach_idx').cumcount()
    mean_ts = df.groupby('time_idx')[column].mean().values
    return mean_ts


def _create_within_session_plot(session, stim, velocity_ts, acceleration_ts, max_accel_idx, max_vel_idx,
                                 energy_ts, energy_min_idx, energy_max_idx,
                                 energy_best_idx, energy_best_dist, energy_best_type,
                                 firing_ts, firing_min_idx, firing_max_idx,
                                 firing_best_idx, firing_best_dist, firing_best_type,
                                 winner, covariance, correlation, save_dir):
    """Create and save a within-session comparison plot with covariance info."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Determine winner color for title
    if winner == 'energy':
        winner_color = 'green'
        winner_text = 'ENERGY WINS'
    elif winner == 'firing':
        winner_color = 'red'
        winner_text = 'FIRING RATE WINS'
    else:
        winner_color = 'gray'
        winner_text = 'TIE'
    
    # Main title with covariance info
    cov_str = f'{covariance:.4f}' if not np.isnan(covariance) else 'N/A'
    corr_str = f'{correlation:.4f}' if not np.isnan(correlation) else 'N/A'
    
    fig.suptitle(f'Stim {stim} | Session: {session} | {winner_text}\n'
                 f'Energy dist: {energy_best_dist} ({energy_best_type}) | '
                 f'Firing dist: {firing_best_dist} ({firing_best_type})\n'
                 f'Covariance(E,FR): {cov_str} | Correlation(E,FR): {corr_str}',
                 fontsize=13, fontweight='bold', color=winner_color)
    
    # -------------------------------------------------------------------------
    # Top plot: Velocity
    # -------------------------------------------------------------------------
    ax = axes[0]
    time_points = np.arange(len(velocity_ts))
    
    ax.plot(time_points, velocity_ts, 'b-', linewidth=1.5, label='Velocity (x)')
    ax.axvline(x=max_vel_idx, color='blue', linestyle='--', linewidth=2, 
               label=f'Max Velocity (idx={max_vel_idx})')
    
    if max_vel_idx < len(velocity_ts):
        ax.plot(max_vel_idx, velocity_ts[max_vel_idx], 'b*', 
                markersize=15, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_ylabel('Velocity_x', fontsize=11, fontweight='bold')
    ax.set_title(f'Session: {session} - Velocity', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # 2nd plot: acceleration
    # -------------------------------------------------------------------------
    ax = axes[1]
    time_points = np.arange(len(acceleration_ts))
    
    ax.plot(time_points, acceleration_ts, 'b-', linewidth=1.5, label='acceleration (x)')
    ax.axvline(x=max_accel_idx, color='blue', linestyle='--', linewidth=2, 
               label=f'Max Acceleration (idx={max_accel_idx})')
    
    if max_accel_idx < len(acceleration_ts):
        ax.plot(max_accel_idx, acceleration_ts[max_accel_idx], 'b*', 
                markersize=15, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_ylabel('Acceleration', fontsize=11, fontweight='bold')
    ax.set_title(f'Session: {session} - Acceleration', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Middle plot: Energy
    # -------------------------------------------------------------------------
    ax = axes[2]
    time_points = np.arange(len(energy_ts))
    
    ax.plot(time_points, energy_ts, 'purple', linewidth=1.5, label='Energy')
    ax.axvline(x=max_accel_idx, color='blue', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Max Vel (idx={max_vel_idx})')
    
    # Mark energy min and max
    if energy_min_idx < len(energy_ts):
        ax.plot(energy_min_idx, energy_ts[energy_min_idx], 'v',
                color='green', markersize=12, markeredgecolor='black', markeredgewidth=1,
                label=f'Energy Min (idx={energy_min_idx})')
    
    if energy_max_idx < len(energy_ts):
        ax.plot(energy_max_idx, energy_ts[energy_max_idx], '^',
                color='red', markersize=12, markeredgecolor='black', markeredgewidth=1,
                label=f'Energy Max (idx={energy_max_idx})')
    
    # Highlight the closest extremum
    if energy_best_idx < len(energy_ts):
        ax.axvline(x=energy_best_idx, color='green', linestyle='-', linewidth=3, alpha=0.5)
    
    ax.annotate(f'Distance: {energy_best_dist} pts\n(closest: {energy_best_type})',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    ax.set_ylabel('Energy', fontsize=11, fontweight='bold')
    ax.set_title(f'Session: {session} - Energy', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Bottom plot: Firing Rate
    # -------------------------------------------------------------------------
    ax = axes[3]
    time_points = np.arange(len(firing_ts))
    
    ax.plot(time_points, firing_ts, 'orange', linewidth=1.5, label='Firing Rate')
    ax.axvline(x=max_accel_idx, color='blue', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Max Vel (idx={max_vel_idx})')
    
    # Mark firing rate min and max
    if firing_min_idx < len(firing_ts):
        ax.plot(firing_min_idx, firing_ts[firing_min_idx], 'v',
                color='green', markersize=12, markeredgecolor='black', markeredgewidth=1,
                label=f'FR Min (idx={firing_min_idx})')
    
    if firing_max_idx < len(firing_ts):
        ax.plot(firing_max_idx, firing_ts[firing_max_idx], '^',
                color='red', markersize=12, markeredgecolor='black', markeredgewidth=1,
                label=f'FR Max (idx={firing_max_idx})')
    
    # Highlight the closest extremum
    if firing_best_idx < len(firing_ts):
        ax.axvline(x=firing_best_idx, color='orange', linestyle='-', linewidth=3, alpha=0.5)
    
    ax.annotate(f'Distance: {firing_best_dist} pts\n(closest: {firing_best_type})',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
    
    ax.set_ylabel('Firing Rate', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Index', fontsize=11, fontweight='bold')
    ax.set_title(f'Session: {session} - Firing Rate', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'stim{stim}_{session}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# # Example usage
# if __name__ == "__main__":
#     np.random.seed(42)
    
#     # Create mock data with original_data dataframes
#     stim_sessions_extrema = {}
    
#     for stim in range(3):
#         stim_sessions_extrema[stim] = {}
        
#         for session in range(5):
#             session_id = f"session_{session}"
            
#             base_vel_max = 400 + np.random.randint(-20, 20)
            
#             # Create mock original_data dataframe
#             n_reaches = 5
#             n_timepoints = 500
            
#             data = []
#             for reach in range(n_reaches):
#                 for t in range(n_timepoints):
#                     data.append({
#                         'reach_idx': reach,
#                         'stim': stim,
#                         'x': np.sin(t / 50) + np.random.randn() * 0.1,
#                         'y': np.cos(t / 50) + np.random.randn() * 0.1,
#                         'z': t / n_timepoints + np.random.randn() * 0.1,
#                         'firing_rate': 10 + 5 * np.sin(t / 30) + np.random.randn(),
#                         'energy': -5 + 2 * np.cos(t / 40) + np.random.randn() * 0.5,
#                         'j': 0.1 + np.random.randn() * 0.01,
#                         'h': 0.05 + np.random.randn() * 0.01
#                     })
            
#             original_data = pd.DataFrame(data)
            
#             # Energy extrema
#             energy_min = base_vel_max + np.random.randint(-15, 15)
#             energy_max = base_vel_max + np.random.randint(-15, 15)
            
#             # Firing rate extrema
#             firing_min = base_vel_max + np.random.randint(-25, 25)
#             firing_max = base_vel_max + np.random.randint(-25, 25)
            
#             stim_sessions_extrema[stim][session_id] = {
#                 'velocity': (base_vel_max - 100, base_vel_max),
#                 'energy': (energy_min, energy_max),
#                 'firing_rate': (firing_min, firing_max),
#                 'original_data': original_data
#             }
    
#     # Run the test with plots
#     print("Running within-session test with plots and covariance...\n")
#     output_dir = '/tmp/within_session_plots'
#     results = within_session_test_with_plots(stim_sessions_extrema, output_dir, verbose=True)
    
#     print(f"\nPlots and summary saved to: {output_dir}")