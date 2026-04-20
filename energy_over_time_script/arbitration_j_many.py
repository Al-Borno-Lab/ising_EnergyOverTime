#!/usr/bin/env python
# coding: utf-8
"""
Batch analysis: do jumps in J (mean neural coupling) align with peak kinematics?

Question answered, session by session:
    "Does the largest rapid change (jump) in J occur near the peak acceleration
     or peak velocity?  Does J lead or lag the kinematic event?"

Run from the energy_over_time_script directory:

    python arbitration_j_many.py --data_folder /path/to/results/

See --help for all options.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.processing import calculate_session_averages, find_extrema_in_range
from src.util import find_file_recursive


# ---------------------------------------------------------------------------
# Session processing (adds j extrema)
# ---------------------------------------------------------------------------

def process_session_j(args):
    """Like process_session but also returns j extrema."""
    stim, session, data, window = args

    stats = calculate_session_averages(data, confidence=0.8)
    stim_stats = stats['by_stimulus'][stim]

    data_frame = {
        'acceleration': find_extrema_in_range(
            stim_stats['acceleration_x']['time_series_mean'], window[0], window[1]),
        'velocity':     find_extrema_in_range(
            stim_stats['velocity_x']['time_series_mean'],     window[0], window[1]),
        'firing_rate':  find_extrema_in_range(
            stim_stats['firing_rate']['time_series_mean'],    window[0], window[1]),
        'energy':       find_extrema_in_range(
            stim_stats['energy']['time_series_mean'],         window[0], window[1]),
        'j':            find_extrema_in_range(
            stim_stats['j']['time_series_mean'],              window[0], window[1]),
        'original_data': data,
    }

    print(f"✓ Done: stim={stim}, session={session}")
    return stim, session, data_frame


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------

def _mean_timeseries(df, col):
    """Mean time series of *col* across reaches (rows grouped by position)."""
    ts = []
    for _, group in df.groupby('reach_idx'):
        ts.append(group[col].values)
    if not ts:
        return np.array([])
    min_len = min(len(a) for a in ts)
    return np.mean([a[:min_len] for a in ts], axis=0)


def _velocity_timeseries(df):
    """Mean x-velocity time series."""
    vels = []
    for _, group in df.groupby('reach_idx'):
        x = group['x'].values
        vels.append(np.diff(x, prepend=x[0]))
    if not vels:
        return np.array([])
    min_len = min(len(v) for v in vels)
    return np.mean([v[:min_len] for v in vels], axis=0)


def _acceleration_timeseries(df):
    vel = _velocity_timeseries(df)
    return np.diff(vel, prepend=vel[0]) if len(vel) else np.array([])


def _xcorr_lag(ts1: np.ndarray, ts2: np.ndarray) -> int:
    """
    Return the lag (in samples) of ts1 relative to ts2 at the cross-correlation peak.

    Sign convention (causal interpretation):
        lag > 0  →  ts1 *leads*  ts2  (ts1 changes before ts2)
        lag < 0  →  ts1 *lags*   ts2  (ts2 changes before ts1)
        lag = 0  →  simultaneous

    Uses mean-subtracted signals so DC offset doesn't bias the result.
    """
    if len(ts1) < 2 or len(ts2) < 2:
        return 0
    n = min(len(ts1), len(ts2))
    a = ts1[:n] - ts1[:n].mean()
    b = ts2[:n] - ts2[:n].mean()
    xcorr = np.correlate(a, b, mode='full')
    # centre of the output corresponds to zero-lag at index n-1
    return int(np.argmax(xcorr)) - (n - 1)


def _j_jump_index(j_ts: np.ndarray, w_lo: int, w_hi: int):
    """
    Find the index (in full time-series coordinates) of the largest rapid change
    in J within the window [w_lo, w_hi].

    A 'jump' is defined as the point of maximum |dJ/dt| — where coupling
    changes fastest.  This is more physically meaningful than a simple extremum
    when looking for coupling transitions linked to movement events.

    Returns
    -------
    jump_idx : int   — index in the full time series
    jump_mag : float — magnitude of the change at that point (|ΔJ|)
    """
    w_lo = max(0, int(w_lo))
    w_hi = min(len(j_ts), int(w_hi))
    if w_hi <= w_lo + 1:
        return w_lo, 0.0
    dj = np.abs(np.diff(j_ts[w_lo:w_hi]))
    local_idx = int(np.argmax(dj))
    return w_lo + local_idx, float(dj[local_idx])


# ---------------------------------------------------------------------------
# Core analysis: J jumps vs kinematic peaks
# ---------------------------------------------------------------------------

def within_session_j_kinematics(stim_sessions_extrema, window, reference="acceleration",
                                 verbose=True):
    """
    For every session: find the largest J jump (max |dJ/dt| in window) and compare
    its timing to both the acceleration peak and the velocity peak.

    Metrics computed per session:
      - j_jump_idx     : time-bin of largest |ΔJ| in window
      - j_jump_mag     : magnitude of that change
      - accel_peak_idx : time-bin of max acceleration in window
      - vel_peak_idx   : time-bin of max velocity in window
      - lag_to_accel   : j_jump_idx − accel_peak_idx  (neg = J leads accel)
      - lag_to_vel     : j_jump_idx − vel_peak_idx    (neg = J leads vel)
      - dist_to_accel  : |lag_to_accel|
      - dist_to_vel    : |lag_to_vel|
      - corr_j_accel   : Pearson r(J ts, accel ts)
      - corr_j_vel     : Pearson r(J ts, vel ts)
      - xcorr_lag_accel: cross-corr peak lag J vs accel (pos = J leads)
      - xcorr_lag_vel  : cross-corr peak lag J vs vel   (pos = J leads)
      - timing_accel   : 'leads' | 'lags' | 'simultaneous'

    Prints a formatted session-by-session table per stimulus.
    """
    results = {'by_stimulus': {}, 'overall': {}, 'all_sessions': []}
    w_lo, w_hi = window

    total_sessions = 0
    all_lag_accel   = []
    all_lag_vel     = []
    all_dist_accel  = []
    all_dist_vel    = []
    all_corr_accel  = []
    all_corr_vel    = []
    all_xcorr_accel = []
    all_xcorr_vel   = []

    for stim in sorted(stim_sessions_extrema):
        sessions = stim_sessions_extrema[stim]
        n_sess   = len(sessions)
        total_sessions += n_sess

        stim_lag_accel   = []
        stim_lag_vel     = []
        stim_dist_accel  = []
        stim_dist_vel    = []
        stim_corr_accel  = []
        stim_corr_vel    = []
        stim_xcorr_accel = []
        stim_xcorr_vel   = []
        session_rows     = []

        if verbose:
            sep = '═' * 78
            print(f"\n{sep}")
            print(f"  STIMULUS {stim}   ({n_sess} sessions)")
            print(sep)
            hdr = (f"  {'Session':<10} {'J_jump':>7} {'|ΔJ|':>8} "
                   f"{'Accel_pk':>9} {'Vel_pk':>8} "
                   f"{'Lag→A':>7} {'Lag→V':>7} "
                   f"{'r(J,A)':>8} {'r(J,V)':>8} "
                   f"{'XC_A':>6} {'XC_V':>6}  Timing")
            print(hdr)
            print('  ' + '-' * (len(hdr) - 2))

        for session, sdata in sessions.items():
            accel_peak_idx = int(sdata['acceleration'][1])
            vel_peak_idx   = int(sdata['velocity'][1])

            # Defaults in case original_data is missing
            j_jump_idx  = w_lo
            j_jump_mag  = np.nan
            corr_j_a    = np.nan
            corr_j_v    = np.nan
            xcorr_a     = np.nan
            xcorr_v     = np.nan

            if 'original_data' in sdata:
                od = sdata['original_data']
                od_stim = od[od['stim'] == stim]
                if len(od_stim) > 0:
                    j_ts     = _mean_timeseries(od_stim, 'j')
                    accel_ts = _acceleration_timeseries(od_stim)
                    vel_ts   = _velocity_timeseries(od_stim)

                    # J jump: index of largest |dJ/dt| in window
                    if len(j_ts) > 0:
                        j_jump_idx, j_jump_mag = _j_jump_index(j_ts, w_lo, w_hi)

                    # Pearson correlations
                    n_ja = min(len(j_ts), len(accel_ts))
                    if n_ja > 1:
                        corr_j_a = float(np.corrcoef(j_ts[:n_ja], accel_ts[:n_ja])[0, 1])
                        stim_corr_accel.append(corr_j_a)
                        all_corr_accel.append(corr_j_a)

                    n_jv = min(len(j_ts), len(vel_ts))
                    if n_jv > 1:
                        corr_j_v = float(np.corrcoef(j_ts[:n_jv], vel_ts[:n_jv])[0, 1])
                        stim_corr_vel.append(corr_j_v)
                        all_corr_vel.append(corr_j_v)

                    # Cross-correlation lags
                    n_ref_a = min(len(j_ts), len(accel_ts))
                    if n_ref_a > 1:
                        xcorr_a = float(_xcorr_lag(j_ts[:n_ref_a], accel_ts[:n_ref_a]))
                        stim_xcorr_accel.append(xcorr_a)
                        all_xcorr_accel.append(xcorr_a)

                    n_ref_v = min(len(j_ts), len(vel_ts))
                    if n_ref_v > 1:
                        xcorr_v = float(_xcorr_lag(j_ts[:n_ref_v], vel_ts[:n_ref_v]))
                        stim_xcorr_vel.append(xcorr_v)
                        all_xcorr_vel.append(xcorr_v)

            lag_to_accel  = j_jump_idx - accel_peak_idx
            lag_to_vel    = j_jump_idx - vel_peak_idx
            dist_to_accel = abs(lag_to_accel)
            dist_to_vel   = abs(lag_to_vel)

            stim_lag_accel.append(lag_to_accel)
            stim_lag_vel.append(lag_to_vel)
            stim_dist_accel.append(dist_to_accel)
            stim_dist_vel.append(dist_to_vel)
            all_lag_accel.append(lag_to_accel)
            all_lag_vel.append(lag_to_vel)
            all_dist_accel.append(dist_to_accel)
            all_dist_vel.append(dist_to_vel)

            # Timing relative to reference signal
            ref_lag = lag_to_accel if reference == 'acceleration' else lag_to_vel
            timing  = 'leads' if ref_lag < 0 else ('lags' if ref_lag > 0 else 'simultaneous')

            row = {
                'stimulus':       stim,
                'session':        session,
                'j_jump_idx':     j_jump_idx,
                'j_jump_mag':     j_jump_mag,
                'accel_peak_idx': accel_peak_idx,
                'vel_peak_idx':   vel_peak_idx,
                'lag_to_accel':   lag_to_accel,
                'lag_to_vel':     lag_to_vel,
                'dist_to_accel':  dist_to_accel,
                'dist_to_vel':    dist_to_vel,
                'corr_j_accel':   corr_j_a,
                'corr_j_vel':     corr_j_v,
                'xcorr_lag_accel': xcorr_a,
                'xcorr_lag_vel':   xcorr_v,
                'timing':         timing,
            }
            session_rows.append(row)
            results['all_sessions'].append(row)

            if verbose:
                na  = lambda v: f"{v:+.3f}" if not np.isnan(v) else "   n/a"
                nai = lambda v: f"{v:+.0f}"  if not np.isnan(v) else "  n/a"
                mag_s = f"{j_jump_mag:.4f}" if not np.isnan(j_jump_mag) else "   n/a"
                print(
                    f"  {session:<10} {j_jump_idx:>7d} {mag_s:>8} "
                    f"{accel_peak_idx:>9d} {vel_peak_idx:>8d} "
                    f"{lag_to_accel:>+7d} {lag_to_vel:>+7d} "
                    f"{na(corr_j_a):>8} {na(corr_j_v):>8} "
                    f"{nai(xcorr_a):>6} {nai(xcorr_v):>6}  {timing}"
                )

        # Stim-level summary
        n_leads = sum(1 for r in session_rows if r['timing'] == 'leads')
        n_lags  = sum(1 for r in session_rows if r['timing'] == 'lags')
        n_simul = sum(1 for r in session_rows if r['timing'] == 'simultaneous')

        results['by_stimulus'][stim] = {
            'n_sessions':        n_sess,
            'n_j_leads':         n_leads,
            'n_j_lags':          n_lags,
            'n_simultaneous':    n_simul,
            'mean_lag_accel':    np.mean(stim_lag_accel)    if stim_lag_accel   else np.nan,
            'std_lag_accel':     np.std(stim_lag_accel)     if stim_lag_accel   else np.nan,
            'mean_lag_vel':      np.mean(stim_lag_vel)      if stim_lag_vel     else np.nan,
            'std_lag_vel':       np.std(stim_lag_vel)       if stim_lag_vel     else np.nan,
            'mean_dist_accel':   np.mean(stim_dist_accel)   if stim_dist_accel  else np.nan,
            'mean_dist_vel':     np.mean(stim_dist_vel)     if stim_dist_vel    else np.nan,
            'mean_corr_j_accel': np.nanmean(stim_corr_accel)  if stim_corr_accel  else np.nan,
            'mean_corr_j_vel':   np.nanmean(stim_corr_vel)    if stim_corr_vel    else np.nan,
            'mean_xcorr_accel':  np.nanmean(stim_xcorr_accel) if stim_xcorr_accel else np.nan,
            'mean_xcorr_vel':    np.nanmean(stim_xcorr_vel)   if stim_xcorr_vel   else np.nan,
            'sessions':          session_rows,
        }

        if verbose:
            s = results['by_stimulus'][stim]
            lag_a_dir = 'leads' if s['mean_lag_accel'] < 0 else 'lags'
            lag_v_dir = 'leads' if s['mean_lag_vel']   < 0 else 'lags'
            print(f"\n  Summary: {n_leads} lead  |  {n_lags} lag  |  {n_simul} simultaneous")
            print(f"  Mean lag → accel: {s['mean_lag_accel']:+.1f} ± {s['std_lag_accel']:.1f} bins  "
                  f"(J {lag_a_dir})")
            print(f"  Mean lag → vel:   {s['mean_lag_vel']:+.1f} ± {s['std_lag_vel']:.1f} bins  "
                  f"(J {lag_v_dir})")
            print(f"  Mean Corr(J, accel): {s['mean_corr_j_accel']:.3f}"
                  f"  |  XCorr lag: {s['mean_xcorr_accel']:+.1f} bins")
            print(f"  Mean Corr(J, vel):   {s['mean_corr_j_vel']:.3f}"
                  f"  |  XCorr lag: {s['mean_xcorr_vel']:+.1f} bins")

    results['overall'] = {
        'n_sessions':        total_sessions,
        'mean_lag_accel':    np.mean(all_lag_accel)    if all_lag_accel   else np.nan,
        'std_lag_accel':     np.std(all_lag_accel)     if all_lag_accel   else np.nan,
        'mean_lag_vel':      np.mean(all_lag_vel)      if all_lag_vel     else np.nan,
        'std_lag_vel':       np.std(all_lag_vel)       if all_lag_vel     else np.nan,
        'mean_dist_accel':   np.mean(all_dist_accel)   if all_dist_accel  else np.nan,
        'mean_dist_vel':     np.mean(all_dist_vel)     if all_dist_vel    else np.nan,
        'mean_corr_j_accel': np.nanmean(all_corr_accel)  if all_corr_accel  else np.nan,
        'mean_corr_j_vel':   np.nanmean(all_corr_vel)    if all_corr_vel    else np.nan,
        'mean_xcorr_accel':  np.nanmean(all_xcorr_accel) if all_xcorr_accel else np.nan,
        'mean_xcorr_vel':    np.nanmean(all_xcorr_vel)   if all_xcorr_vel   else np.nan,
    }

    if verbose:
        ov = results['overall']
        n_leads_all = sum(1 for r in results['all_sessions'] if r['timing'] == 'leads')
        n_lags_all  = sum(1 for r in results['all_sessions'] if r['timing'] == 'lags')
        n_simul_all = sum(1 for r in results['all_sessions'] if r['timing'] == 'simultaneous')
        lag_a_dir = 'leads' if ov['mean_lag_accel'] < 0 else 'lags'
        lag_v_dir = 'leads' if ov['mean_lag_vel']   < 0 else 'lags'
        print(f"\n{'═'*78}")
        print(f"  OVERALL  ({total_sessions} sessions)")
        print(f"{'═'*78}")
        print(f"  J leads: {n_leads_all}  |  J lags: {n_lags_all}  |  simultaneous: {n_simul_all}")
        print(f"  Mean lag → accel: {ov['mean_lag_accel']:+.1f} ± {ov['std_lag_accel']:.1f} bins  "
              f"(J {lag_a_dir})")
        print(f"  Mean lag → vel:   {ov['mean_lag_vel']:+.1f} ± {ov['std_lag_vel']:.1f} bins  "
              f"(J {lag_v_dir})")
        print(f"  Mean Corr(J, accel): {ov['mean_corr_j_accel']:.3f}"
              f"  |  XCorr lag: {ov['mean_xcorr_accel']:+.1f} bins")
        print(f"  Mean Corr(J, vel):   {ov['mean_corr_j_vel']:.3f}"
              f"  |  XCorr lag: {ov['mean_xcorr_vel']:+.1f} bins")

    return results


# ---------------------------------------------------------------------------
# Per-session plot: velocity | acceleration | J with jump marker
# ---------------------------------------------------------------------------

def _plot_session_j_kinematics(session, stim, reference,
                                accel_ts, vel_ts, j_ts,
                                j_jump_idx, j_jump_mag,
                                accel_peak_idx, vel_peak_idx,
                                lag_to_accel, lag_to_vel,
                                corr_j_accel, corr_j_vel,
                                xcorr_lag_accel, xcorr_lag_vel,
                                timing, save_dir):
    """3-panel figure: velocity | acceleration | J coupling with jump marked."""

    na_fmt  = lambda v: f"{v:.3f}" if not np.isnan(v) else "n/a"
    lag_dir = timing   # 'leads' | 'lags' | 'simultaneous'

    panel_color = {'leads': 'limegreen', 'lags': 'tomato',
                   'simultaneous': 'gold'}[timing]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"Session {session}  |  Stim {stim}  |  J {lag_dir.upper()} kinematics\n"
        f"J jump idx={j_jump_idx}  |ΔJ|={na_fmt(j_jump_mag)}\n"
        f"Accel peak={accel_peak_idx} (lag {lag_to_accel:+d})  "
        f"Vel peak={vel_peak_idx} (lag {lag_to_vel:+d})\n"
        f"Corr(J,accel)={na_fmt(corr_j_accel)}  XClag={xcorr_lag_accel:+.0f}  |  "
        f"Corr(J,vel)={na_fmt(corr_j_vel)}  XClag={xcorr_lag_vel:+.0f}",
        fontsize=10, fontweight='bold'
    )

    # ── Panel 1: Velocity ──────────────────────────────────────────────────
    axes[0].plot(vel_ts, color='navy', linewidth=1.5, label='X velocity')
    axes[0].axvline(vel_peak_idx, color='deepskyblue', linestyle='--', linewidth=2,
                    label=f'Vel peak (idx={vel_peak_idx})')
    axes[0].axvline(j_jump_idx, color='crimson', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'J jump (idx={j_jump_idx})')
    axes[0].set_ylabel("X Velocity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # ── Panel 2: Acceleration ─────────────────────────────────────────────
    axes[1].plot(accel_ts, color='steelblue', linewidth=1.5, label='X acceleration')
    axes[1].axvline(accel_peak_idx, color='green', linestyle='--', linewidth=2,
                    label=f'Accel peak (idx={accel_peak_idx})')
    axes[1].axvline(j_jump_idx, color='crimson', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'J jump (idx={j_jump_idx})')
    axes[1].set_ylabel("X Acceleration")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    # ── Panel 3: J + dJ/dt with jump highlighted ──────────────────────────
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(j_ts, color='darkorchid', linewidth=1.5, label='J (mean coupling)')
    ax3.axvline(j_jump_idx, color='crimson', linewidth=2.5,
                label=f'J jump (|ΔJ|={na_fmt(j_jump_mag)})')
    ax3.axvline(accel_peak_idx, color='green',      linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Accel peak')
    ax3.axvline(vel_peak_idx,   color='deepskyblue', linestyle='--', linewidth=1.5, alpha=0.6,
                label=f'Vel peak')
    ax3.set_facecolor((*mcolors.to_rgb(panel_color), 0.10))
    ax3.set_ylabel("J (coupling)", color='darkorchid')
    ax3.tick_params(axis='y', labelcolor='darkorchid')

    if len(j_ts) > 1:
        dj = np.abs(np.diff(j_ts, prepend=j_ts[0]))
        ax3_twin.plot(dj, color='salmon', linewidth=1.0, alpha=0.6, linestyle='-',
                      label='|dJ/dt|')
        ax3_twin.set_ylabel("|dJ/dt|", color='salmon')
        ax3_twin.tick_params(axis='y', labelcolor='salmon')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    ax3.set_xlabel("Time bin")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"stim{stim}_{session}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator: analysis + plots + CSV
# ---------------------------------------------------------------------------

def j_kinematics_with_plots(stim_sessions_extrema, window, output_dir,
                             reference="acceleration", verbose=True):
    """
    Run J-jump vs kinematic peak analysis; save per-session plots and CSVs.

    Plots are saved into sub-directories by timing relationship:
        leads/        — J jump precedes the kinematic peak
        lags/         — J jump follows the kinematic peak
        simultaneous/ — J jump and kinematic peak coincide
    """
    leads_dir  = os.path.join(output_dir, 'j_leads')
    lags_dir   = os.path.join(output_dir, 'j_lags')
    simul_dir  = os.path.join(output_dir, 'simultaneous')
    for d in (leads_dir, lags_dir, simul_dir):
        os.makedirs(d, exist_ok=True)

    results = within_session_j_kinematics(
        stim_sessions_extrema, window=window, reference=reference, verbose=verbose
    )

    for row in results['all_sessions']:
        stim    = row['stimulus']
        session = row['session']
        sdata   = stim_sessions_extrema[stim].get(session, {})

        if 'original_data' not in sdata:
            continue
        od = sdata['original_data'][sdata['original_data']['stim'] == stim]
        if len(od) == 0:
            continue

        accel_ts = _acceleration_timeseries(od)
        vel_ts   = _velocity_timeseries(od)
        j_ts     = _mean_timeseries(od, 'j')

        timing   = row['timing']
        save_dir = leads_dir if timing == 'leads' else (lags_dir if timing == 'lags' else simul_dir)

        xcorr_a = row['xcorr_lag_accel'] if not np.isnan(row['xcorr_lag_accel']) else 0
        xcorr_v = row['xcorr_lag_vel']   if not np.isnan(row['xcorr_lag_vel'])   else 0

        _plot_session_j_kinematics(
            session=session, stim=stim, reference=reference,
            accel_ts=accel_ts, vel_ts=vel_ts, j_ts=j_ts,
            j_jump_idx=row['j_jump_idx'], j_jump_mag=row['j_jump_mag'],
            accel_peak_idx=row['accel_peak_idx'], vel_peak_idx=row['vel_peak_idx'],
            lag_to_accel=row['lag_to_accel'], lag_to_vel=row['lag_to_vel'],
            corr_j_accel=row['corr_j_accel'], corr_j_vel=row['corr_j_vel'],
            xcorr_lag_accel=xcorr_a, xcorr_lag_vel=xcorr_v,
            timing=timing, save_dir=save_dir,
        )

    # Session-level CSV
    summary_df = pd.DataFrame([{
        'stimulus':        r['stimulus'],
        'session':         r['session'],
        'j_jump_idx':      r['j_jump_idx'],
        'j_jump_mag':      r['j_jump_mag'],
        'accel_peak_idx':  r['accel_peak_idx'],
        'vel_peak_idx':    r['vel_peak_idx'],
        'lag_to_accel':    r['lag_to_accel'],
        'lag_to_vel':      r['lag_to_vel'],
        'dist_to_accel':   r['dist_to_accel'],
        'dist_to_vel':     r['dist_to_vel'],
        'corr_j_accel':    r['corr_j_accel'],
        'corr_j_vel':      r['corr_j_vel'],
        'xcorr_lag_accel': r['xcorr_lag_accel'],
        'xcorr_lag_vel':   r['xcorr_lag_vel'],
        'timing':          r['timing'],
    } for r in results['all_sessions']])
    summary_df.to_csv(os.path.join(output_dir, 'session_summary_j_kinematics.csv'), index=False)

    n_leads = (summary_df['timing'] == 'leads').sum()
    n_lags  = (summary_df['timing'] == 'lags').sum()
    n_sim   = (summary_df['timing'] == 'simultaneous').sum()

    if verbose:
        print(f"\n  Plots saved:")
        print(f"    {n_leads} J-leads sessions  → {leads_dir}")
        print(f"    {n_lags}  J-lags sessions   → {lags_dir}")
        print(f"    {n_sim}  simultaneous      → {simul_dir}")

    results['summary_df'] = summary_df
    return results


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def extract_stats_to_df(stats_dict, **kwargs):
    s = stats_dict
    row = {
        'n_sessions':        s['n_sessions'],
        'n_j_leads':         s.get('n_j_leads',      np.nan),
        'n_j_lags':          s.get('n_j_lags',       np.nan),
        'n_simultaneous':    s.get('n_simultaneous',  np.nan),
        'mean_lag_accel':    s.get('mean_lag_accel',  np.nan),
        'std_lag_accel':     s.get('std_lag_accel',   np.nan),
        'mean_lag_vel':      s.get('mean_lag_vel',    np.nan),
        'std_lag_vel':       s.get('std_lag_vel',     np.nan),
        'mean_dist_accel':   s.get('mean_dist_accel', np.nan),
        'mean_dist_vel':     s.get('mean_dist_vel',   np.nan),
        'mean_corr_j_accel': s.get('mean_corr_j_accel', np.nan),
        'mean_corr_j_vel':   s.get('mean_corr_j_vel',   np.nan),
        'mean_xcorr_accel':  s.get('mean_xcorr_accel',  np.nan),
        'mean_xcorr_vel':    s.get('mean_xcorr_vel',     np.nan),
        **kwargs,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Path helpers (same as arbitration_many.py)
# ---------------------------------------------------------------------------

def _rep_from_session_path(session):
    return session.split("_results/")[1].split("/")[0].split("_")[1][3:]


def _session_id_from_path(session):
    return session.split("_results")[0][-6:]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_j_kinematics_markdown(all_results, out_base, report_dir, args):
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, "j_kinematics_results.md")

    lines = [
        "# J-Jump vs Kinematic Peak Analysis",
        "",
        f"*Generated: {date.today().isoformat()}*",
        "",
        "**Question:** Do rapid changes (jumps) in J (mean neural coupling) "
        "align with peak kinematic events?  Does J lead or lag the movement?",
        "",
        f"**Data folder:** `{args.data_folder}`  ",
        f"**Window:** [{args.window[0]}, {args.window[1]}]  ",
        f"**Reference signal:** {args.reference}  ",
        f"**Reps:** {args.rep_start} – {args.rep_end_exclusive - 1}  ",
        f"**Stimuli:** {args.stim_min} – {args.stim_max_exclusive - 1}",
        "",
        "> J **jump** = index of max |dJ/dt| within the search window.",
        "> **Lag** = j\\_jump\\_idx − kinematic\\_peak\\_idx  "
        "(negative = J precedes kinematics = J *leads*; positive = J *lags*).",
        "",
        "---",
        "",
        "## Overall Timing Summary (per Rep)",
        "",
        "| Rep | N Sessions | J Leads | J Lags | Simultaneous | "
        "Mean Lag→Accel | Mean Lag→Vel |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        ov = res['overall']
        all_sess = res['all_sessions']
        n_leads = sum(1 for r in all_sess if r['timing'] == 'leads')
        n_lags  = sum(1 for r in all_sess if r['timing'] == 'lags')
        n_sim   = sum(1 for r in all_sess if r['timing'] == 'simultaneous')
        lines.append(
            f"| {rep} | {ov['n_sessions']} | {n_leads} | {n_lags} | {n_sim} | "
            f"{ov['mean_lag_accel']:+.1f} | {ov['mean_lag_vel']:+.1f} |"
        )

    lines += [
        "",
        "## Per-Stimulus Breakdown",
        "",
        "| Rep | Stim | N | Leads | Lags | Simult | "
        "Lag→A (mean±std) | Lag→V (mean±std) | "
        "Corr(J,A) | Corr(J,V) | XClag A | XClag V |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        for stim in sorted(res['by_stimulus']):
            s = res['by_stimulus'][stim]
            lines.append(
                f"| {rep} | {stim} | {s['n_sessions']} | "
                f"{s['n_j_leads']} | {s['n_j_lags']} | {s['n_simultaneous']} | "
                f"{s['mean_lag_accel']:+.1f}±{s['std_lag_accel']:.1f} | "
                f"{s['mean_lag_vel']:+.1f}±{s['std_lag_vel']:.1f} | "
                f"{s['mean_corr_j_accel']:.3f} | {s['mean_corr_j_vel']:.3f} | "
                f"{s['mean_xcorr_accel']:+.1f} | {s['mean_xcorr_vel']:+.1f} |"
            )

    lines += [
        "",
        "## Session-by-Session Detail",
        "",
        "| Rep | Stim | Session | J jump | |ΔJ| | Accel pk | Vel pk | "
        "Lag→A | Lag→V | r(J,A) | r(J,V) | Timing |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        for row in res['all_sessions']:
            mag = f"{row['j_jump_mag']:.4f}" if not np.isnan(row['j_jump_mag']) else "n/a"
            rja = f"{row['corr_j_accel']:.3f}" if not np.isnan(row['corr_j_accel']) else "n/a"
            rjv = f"{row['corr_j_vel']:.3f}"   if not np.isnan(row['corr_j_vel'])   else "n/a"
            lines.append(
                f"| {rep} | {row['stimulus']} | {row['session']} | "
                f"{row['j_jump_idx']} | {mag} | "
                f"{row['accel_peak_idx']} | {row['vel_peak_idx']} | "
                f"{row['lag_to_accel']:+d} | {row['lag_to_vel']:+d} | "
                f"{rja} | {rjv} | **{row['timing']}** |"
            )

    lines += ["", "## Output Files", ""]
    for rep in sorted(all_results):
        rep_out = os.path.join(out_base, str(rep))
        lines += [
            f"**Rep {rep}**  ",
            f"- Session CSV: `{os.path.join(rep_out, 'session_summary_j_kinematics.csv')}`  ",
            f"- Results CSV: `{os.path.join(rep_out, 'results_j_kinematics.csv')}`  ",
            f"- J-leads plots: `{os.path.join(rep_out, 'j_leads')}/`  ",
            f"- J-lags plots:  `{os.path.join(rep_out, 'j_lags')}/`  ",
            f"- Simultaneous:  `{os.path.join(rep_out, 'simultaneous')}/`  ",
            "",
        ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="J-jump vs kinematic peak: does max |dJ/dt| align with "
                    "peak acceleration / velocity?"
    )
    p.add_argument("--data_folder", required=True,
                   help="Root folder to search recursively for per_reach_state.csv")
    p.add_argument("--rep_start", type=int, default=1,
                   help="First rep index (default 1)")
    p.add_argument("--rep_end_exclusive", type=int, default=2,
                   help="One past last rep index (default 2)")
    p.add_argument("--window", type=int, nargs=2, metavar=("LO", "HI"),
                   default=[390, 410],
                   help="Time-index window for jump/peak search (default: 390 410)")
    p.add_argument("--stim_min", type=int, default=0,
                   help="Inclusive minimum stimulus index (default 0)")
    p.add_argument("--stim_max_exclusive", type=int, default=3,
                   help="Exclusive max stimulus index (default 3)")
    p.add_argument("--output_base", type=str, default=None,
                   help="Base directory for outputs")
    p.add_argument("--reference", choices=("acceleration", "velocity"),
                   default="acceleration",
                   help="Primary kinematic reference for timing classification "
                        "(default: acceleration)")
    p.add_argument("--workers", type=int, default=None,
                   help="Pool worker count (default: multiprocessing default)")
    p.add_argument("--quiet_find", action="store_true",
                   help="Less verbose output from find_file_recursive")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-session table output")
    p.add_argument("--report_dir", type=str, default=None, metavar="DIR",
                   help="If set, write j_kinematics_results.md into this directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    data_folder = os.path.abspath(os.path.expanduser(args.data_folder))
    if not os.path.isdir(data_folder):
        print(f"Error: {data_folder} is not a directory", file=sys.stderr)
        return 1

    w_lo, w_hi = args.window
    if args.output_base is None:
        out_base = os.path.abspath(
            f"./Arbitration/j_kinematics/w{w_lo}_{w_hi}"
        )
    else:
        out_base = os.path.abspath(os.path.expanduser(args.output_base))

    all_reach_states = find_file_recursive(
        data_folder, "per_reach_state.csv", verbose=not args.quiet_find)
    if not all_reach_states:
        print(f"No per_reach_state.csv found under {data_folder}", file=sys.stderr)
        return 1

    stim_range = range(args.stim_min, args.stim_max_exclusive)
    window     = [w_lo, w_hi]
    all_results = {}

    for rep in range(args.rep_start, args.rep_end_exclusive):
        session_data = {}
        rep_s = str(rep)

        for session in all_reach_states:
            if "full_reach" not in session:
                continue
            try:
                if _rep_from_session_path(session) != rep_s:
                    continue
            except (IndexError, ValueError) as e:
                print(f"Skipping (could not parse rep): {session} ({e})", file=sys.stderr)
                continue

            session_id = _session_id_from_path(session)
            print(session)
            print(f"Session_id: {session_id}")
            session_data[session_id] = pd.read_csv(session)

        if not session_data:
            print(f"No sessions matched rep={rep} and full_reach", file=sys.stderr)
            continue

        tasks = [
            (stim, session, session_data[session], window)
            for stim in stim_range
            for session in session_data
        ]

        stim_sessions_extrema = {}
        pool_kw = {} if args.workers is None else {'processes': args.workers}
        with Pool(**pool_kw) as pool:
            for stim, session, df in pool.imap_unordered(process_session_j, tasks):
                print(f"← Received: stim={stim}, session={session}")
                stim_sessions_extrema.setdefault(stim, {})[session] = df

        print("All done!")

        rep_out = os.path.join(out_base, str(rep))
        os.makedirs(rep_out, exist_ok=True)

        results = j_kinematics_with_plots(
            stim_sessions_extrema,
            window=window,
            output_dir=rep_out,
            reference=args.reference,
            verbose=not args.quiet,
        )
        all_results[rep] = results

        df_out = pd.concat(
            [extract_stats_to_df(results['by_stimulus'][stim], stim=stim, rep=rep)
             for stim in results['by_stimulus']],
            ignore_index=True,
        )
        df_out.to_csv(os.path.join(rep_out, "results_j_kinematics.csv"), index=False)
        print(f"Wrote {os.path.join(rep_out, 'results_j_kinematics.csv')}")

    if args.report_dir and all_results:
        write_j_kinematics_markdown(all_results, out_base, args.report_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
