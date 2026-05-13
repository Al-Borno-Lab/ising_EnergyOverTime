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
from src.peak_detection import (
    gaussian_smooth   as _gaussian_smooth,
    local_prominence  as _local_prominence,
    detect_j_peak     as _j_peak_detect,
    detect_signal_extremum as _signal_extremum_detect,
)


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
                                 peak_threshold=1.5, smooth_sigma=5.0, verbose=True):
    """
    For every session: find the largest J jump (max |dJ/dt| in window) and compare
    its timing to both the acceleration peak and the velocity peak.
    Also tests whether J has a large peak in the window and, if so, reports
    its lag relative to each kinematic signal.

    Metrics computed per session:
      - j_jump_idx          : time-bin of largest |ΔJ| in window
      - j_jump_mag          : magnitude of that change
      - accel_peak_idx      : time-bin of max acceleration in window
      - vel_peak_idx        : time-bin of max velocity in window
      - lag_to_accel        : j_jump_idx − accel_peak_idx  (neg = J leads accel)
      - lag_to_vel          : j_jump_idx − vel_peak_idx    (neg = J leads vel)
      - dist_to_accel       : |lag_to_accel|
      - dist_to_vel         : |lag_to_vel|
      - corr_j_accel        : Pearson r(J ts, accel ts)
      - corr_j_vel          : Pearson r(J ts, vel ts)
      - xcorr_lag_accel     : cross-corr peak lag J vs accel (pos = J leads)
      - xcorr_lag_vel       : cross-corr peak lag J vs vel   (pos = J leads)
      - timing_accel        : 'leads' | 'lags' | 'simultaneous'
      - has_j_peak          : bool — J has a large peak in the window
      - j_peak_idx          : time-bin of the J peak (nan if no peak)
      - j_peak_value        : J value at the peak (nan if no peak)
      - j_peak_z            : prom/bg90 ratio (prominence vs background undulations)
      - j_peak_lag_to_accel : j_peak_idx − accel_peak_idx (nan if no peak)
      - j_peak_lag_to_vel   : j_peak_idx − vel_peak_idx   (nan if no peak)

    Prints a formatted session-by-session table per stimulus.
    """
    results = {'by_stimulus': {}, 'overall': {}, 'all_sessions': []}
    w_lo, w_hi = window

    total_sessions   = 0
    all_lag_accel    = []
    all_lag_vel      = []
    all_dist_accel   = []
    all_dist_vel     = []
    all_corr_accel   = []
    all_corr_vel     = []
    all_xcorr_accel  = []
    all_xcorr_vel    = []
    all_has_peak     = []
    # Firing rate lags
    all_fr_min_lag_accel = []
    all_fr_min_lag_vel   = []
    all_fr_max_lag_accel = []
    all_fr_max_lag_vel   = []
    # Energy lags
    all_en_min_lag_accel = []
    all_en_min_lag_vel   = []
    all_en_max_lag_accel = []
    all_en_max_lag_vel   = []

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
        stim_has_peak    = []
        stim_fr_min_lag_accel = []
        stim_fr_min_lag_vel   = []
        stim_fr_max_lag_accel = []
        stim_fr_max_lag_vel   = []
        stim_en_min_lag_accel = []
        stim_en_min_lag_vel   = []
        stim_en_max_lag_accel = []
        stim_en_max_lag_vel   = []
        session_rows     = []

        if verbose:
            sep = '═' * 90
            print(f"\n{sep}")
            print(f"  STIMULUS {stim}   ({n_sess} sessions)")
            print(sep)
            hdr = (f"  {'Session':<10} {'J_jump':>7} {'|ΔJ|':>8} "
                   f"{'Accel_pk':>9} {'Vel_pk':>8} "
                   f"{'Lag→A':>7} {'Lag→V':>7} "
                   f"{'r(J,A)':>8} {'r(J,V)':>8} "
                   f"{'XC_A':>6} {'XC_V':>6}  "
                   f"{'Peak?':>6} {'Ratio':>6} {'PkLag→A':>8} {'PkLag→V':>8}  Timing")
            print(hdr)
            print('  ' + '-' * (len(hdr) - 2))

        for session, sdata in sessions.items():
            accel_peak_idx = int(sdata['acceleration'][1])
            vel_peak_idx   = int(sdata['velocity'][1])

            # Firing rate & energy extrema (min/max within window from process_session_j)
            fr_min_idx = int(sdata.get('firing_rate', (w_lo, w_lo))[0])
            fr_max_idx = int(sdata.get('firing_rate', (w_lo, w_lo))[1])
            en_min_idx = int(sdata.get('energy',      (w_lo, w_lo))[0])
            en_max_idx = int(sdata.get('energy',      (w_lo, w_lo))[1])

            fr_min_lag_accel = fr_min_idx - accel_peak_idx
            fr_min_lag_vel   = fr_min_idx - vel_peak_idx
            fr_max_lag_accel = fr_max_idx - accel_peak_idx
            fr_max_lag_vel   = fr_max_idx - vel_peak_idx

            en_min_lag_accel = en_min_idx - accel_peak_idx
            en_min_lag_vel   = en_min_idx - vel_peak_idx
            en_max_lag_accel = en_max_idx - accel_peak_idx
            en_max_lag_vel   = en_max_idx - vel_peak_idx

            # "Closer to" determination per extremum
            fr_min_closer = 'accel' if abs(fr_min_lag_accel) <= abs(fr_min_lag_vel) else 'vel'
            fr_max_closer = 'accel' if abs(fr_max_lag_accel) <= abs(fr_max_lag_vel) else 'vel'
            en_min_closer = 'accel' if abs(en_min_lag_accel) <= abs(en_min_lag_vel) else 'vel'
            en_max_closer = 'accel' if abs(en_max_lag_accel) <= abs(en_max_lag_vel) else 'vel'

            # Defaults in case original_data is missing
            j_jump_idx  = w_lo
            j_jump_mag  = np.nan
            corr_j_a    = np.nan
            corr_j_v    = np.nan
            xcorr_a     = np.nan
            xcorr_v     = np.nan

            # J peak-detection defaults
            has_j_peak          = False
            j_peak_idx          = np.nan
            j_peak_value        = np.nan
            j_peak_z            = np.nan
            j_peak_lag_to_accel = np.nan
            j_peak_lag_to_vel   = np.nan

            # FR detected peak/trough defaults
            has_fr_peak         = False
            fr_peak_idx         = w_lo
            fr_peak_value       = np.nan
            fr_peak_z           = np.nan
            fr_peak_lag_accel   = np.nan
            fr_peak_lag_vel     = np.nan
            has_fr_trough       = False
            fr_trough_idx       = w_lo
            fr_trough_value     = np.nan
            fr_trough_z         = np.nan
            fr_trough_lag_accel = np.nan
            fr_trough_lag_vel   = np.nan

            # Energy detected peak/trough defaults
            has_en_peak         = False
            en_peak_idx         = w_lo
            en_peak_value       = np.nan
            en_peak_z           = np.nan
            en_peak_lag_accel   = np.nan
            en_peak_lag_vel     = np.nan
            has_en_trough       = False
            en_trough_idx       = w_lo
            en_trough_value     = np.nan
            en_trough_z         = np.nan
            en_trough_lag_accel = np.nan
            en_trough_lag_vel   = np.nan

            if 'original_data' in sdata:
                od = sdata['original_data']
                od_stim = od[od['stim'] == stim]
                if len(od_stim) > 0:
                    j_ts      = _mean_timeseries(od_stim, 'j')
                    accel_ts  = _acceleration_timeseries(od_stim)
                    vel_ts    = _velocity_timeseries(od_stim)
                    fr_ts_det = (_mean_timeseries(od_stim, 'firing_rate')
                                 if 'firing_rate' in od_stim.columns else np.array([]))
                    en_ts_det = (_mean_timeseries(od_stim, 'energy')
                                 if 'energy' in od_stim.columns else np.array([]))

                    if len(j_ts) > 0:
                        # J jump: index of largest |dJ/dt| in window
                        j_jump_idx, j_jump_mag = _j_jump_index(j_ts, w_lo, w_hi)

                        # J peak: prominent max in window (bump + step detection)
                        has_j_peak, _pk_idx, j_peak_value, j_peak_z = \
                            _j_peak_detect(j_ts, w_lo, w_hi,
                                           threshold_ratio=peak_threshold,
                                           smooth_sigma=smooth_sigma)
                        if has_j_peak:
                            j_peak_idx          = int(_pk_idx)
                            j_peak_lag_to_accel = j_peak_idx - accel_peak_idx
                            j_peak_lag_to_vel   = j_peak_idx - vel_peak_idx
                        else:
                            j_peak_idx = int(_pk_idx)   # store anyway for plotting

                    # ── Firing rate peak / trough detection ───────────────
                    if len(fr_ts_det) > 0:
                        has_fr_peak, _fpi, fr_peak_value, fr_peak_z = \
                            _signal_extremum_detect(fr_ts_det, w_lo, w_hi,
                                                    threshold_ratio=peak_threshold,
                                                    smooth_sigma=smooth_sigma,
                                                    kind='peak')
                        fr_peak_idx = int(_fpi)
                        if has_fr_peak:
                            fr_peak_lag_accel = fr_peak_idx - accel_peak_idx
                            fr_peak_lag_vel   = fr_peak_idx - vel_peak_idx

                        has_fr_trough, _fti, fr_trough_value, fr_trough_z = \
                            _signal_extremum_detect(fr_ts_det, w_lo, w_hi,
                                                    threshold_ratio=peak_threshold,
                                                    smooth_sigma=smooth_sigma,
                                                    kind='trough')
                        fr_trough_idx = int(_fti)
                        if has_fr_trough:
                            fr_trough_lag_accel = fr_trough_idx - accel_peak_idx
                            fr_trough_lag_vel   = fr_trough_idx - vel_peak_idx

                    # ── Energy peak / trough detection ────────────────────
                    if len(en_ts_det) > 0:
                        has_en_peak, _epi, en_peak_value, en_peak_z = \
                            _signal_extremum_detect(en_ts_det, w_lo, w_hi,
                                                    threshold_ratio=peak_threshold,
                                                    smooth_sigma=smooth_sigma,
                                                    kind='peak')
                        en_peak_idx = int(_epi)
                        if has_en_peak:
                            en_peak_lag_accel = en_peak_idx - accel_peak_idx
                            en_peak_lag_vel   = en_peak_idx - vel_peak_idx

                        has_en_trough, _eti, en_trough_value, en_trough_z = \
                            _signal_extremum_detect(en_ts_det, w_lo, w_hi,
                                                    threshold_ratio=peak_threshold,
                                                    smooth_sigma=smooth_sigma,
                                                    kind='trough')
                        en_trough_idx = int(_eti)
                        if has_en_trough:
                            en_trough_lag_accel = en_trough_idx - accel_peak_idx
                            en_trough_lag_vel   = en_trough_idx - vel_peak_idx

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
            stim_has_peak.append(has_j_peak)
            stim_fr_min_lag_accel.append(fr_min_lag_accel)
            stim_fr_min_lag_vel.append(fr_min_lag_vel)
            stim_fr_max_lag_accel.append(fr_max_lag_accel)
            stim_fr_max_lag_vel.append(fr_max_lag_vel)
            stim_en_min_lag_accel.append(en_min_lag_accel)
            stim_en_min_lag_vel.append(en_min_lag_vel)
            stim_en_max_lag_accel.append(en_max_lag_accel)
            stim_en_max_lag_vel.append(en_max_lag_vel)
            all_lag_accel.append(lag_to_accel)
            all_lag_vel.append(lag_to_vel)
            all_dist_accel.append(dist_to_accel)
            all_dist_vel.append(dist_to_vel)
            all_has_peak.append(has_j_peak)
            all_fr_min_lag_accel.append(fr_min_lag_accel)
            all_fr_min_lag_vel.append(fr_min_lag_vel)
            all_fr_max_lag_accel.append(fr_max_lag_accel)
            all_fr_max_lag_vel.append(fr_max_lag_vel)
            all_en_min_lag_accel.append(en_min_lag_accel)
            all_en_min_lag_vel.append(en_min_lag_vel)
            all_en_max_lag_accel.append(en_max_lag_accel)
            all_en_max_lag_vel.append(en_max_lag_vel)

            # Timing relative to reference signal
            ref_lag = lag_to_accel if reference == 'acceleration' else lag_to_vel
            timing  = 'leads' if ref_lag < 0 else ('lags' if ref_lag > 0 else 'simultaneous')

            row = {
                'stimulus':             stim,
                'session':              session,
                # Kinematic reference peaks
                'accel_peak_idx':       accel_peak_idx,
                'vel_peak_idx':         vel_peak_idx,
                # Firing rate extrema
                'fr_min_idx':           fr_min_idx,
                'fr_max_idx':           fr_max_idx,
                'fr_min_lag_accel':     fr_min_lag_accel,
                'fr_min_lag_vel':       fr_min_lag_vel,
                'fr_max_lag_accel':     fr_max_lag_accel,
                'fr_max_lag_vel':       fr_max_lag_vel,
                'fr_min_closer':        fr_min_closer,
                'fr_max_closer':        fr_max_closer,
                # Energy extrema
                'en_min_idx':           en_min_idx,
                'en_max_idx':           en_max_idx,
                'en_min_lag_accel':     en_min_lag_accel,
                'en_min_lag_vel':       en_min_lag_vel,
                'en_max_lag_accel':     en_max_lag_accel,
                'en_max_lag_vel':       en_max_lag_vel,
                'en_min_closer':        en_min_closer,
                'en_max_closer':        en_max_closer,
                # J jump
                'j_jump_idx':           j_jump_idx,
                'j_jump_mag':           j_jump_mag,
                'lag_to_accel':         lag_to_accel,
                'lag_to_vel':           lag_to_vel,
                'dist_to_accel':        dist_to_accel,
                'dist_to_vel':          dist_to_vel,
                'corr_j_accel':         corr_j_a,
                'corr_j_vel':           corr_j_v,
                'xcorr_lag_accel':      xcorr_a,
                'xcorr_lag_vel':        xcorr_v,
                'timing':               timing,
                # J peak
                'has_j_peak':           has_j_peak,
                'j_peak_idx':           j_peak_idx,
                'j_peak_value':         j_peak_value,
                'j_peak_z':             j_peak_z,
                'j_peak_lag_to_accel':  j_peak_lag_to_accel,
                'j_peak_lag_to_vel':    j_peak_lag_to_vel,
                # Firing rate detected peak / trough
                'has_fr_peak':          has_fr_peak,
                'fr_peak_idx':          fr_peak_idx,
                'fr_peak_z':            fr_peak_z,
                'fr_peak_lag_accel':    fr_peak_lag_accel,
                'fr_peak_lag_vel':      fr_peak_lag_vel,
                'has_fr_trough':        has_fr_trough,
                'fr_trough_idx':        fr_trough_idx,
                'fr_trough_z':          fr_trough_z,
                'fr_trough_lag_accel':  fr_trough_lag_accel,
                'fr_trough_lag_vel':    fr_trough_lag_vel,
                # Energy detected peak / trough
                'has_en_peak':          has_en_peak,
                'en_peak_idx':          en_peak_idx,
                'en_peak_z':            en_peak_z,
                'en_peak_lag_accel':    en_peak_lag_accel,
                'en_peak_lag_vel':      en_peak_lag_vel,
                'has_en_trough':        has_en_trough,
                'en_trough_idx':        en_trough_idx,
                'en_trough_z':          en_trough_z,
                'en_trough_lag_accel':  en_trough_lag_accel,
                'en_trough_lag_vel':    en_trough_lag_vel,
            }
            session_rows.append(row)
            results['all_sessions'].append(row)

            if verbose:
                na   = lambda v: f"{v:+.3f}" if (isinstance(v, float) and not np.isnan(v)) else "   n/a"
                nai  = lambda v: f"{v:+.0f}"  if (isinstance(v, float) and not np.isnan(v)) else "  n/a"
                naf  = lambda v: f"{v:.2f}"   if (isinstance(v, float) and not np.isnan(v)) else "  n/a"
                mag_s    = f"{j_jump_mag:.4f}" if not np.isnan(j_jump_mag) else "   n/a"
                peak_sym = "YES" if has_j_peak else " no"
                pk_z_s   = naf(j_peak_z)
                pk_la_s  = nai(j_peak_lag_to_accel)
                pk_lv_s  = nai(j_peak_lag_to_vel)
                print(
                    f"  {session:<10} {j_jump_idx:>7d} {mag_s:>8} "
                    f"{accel_peak_idx:>9d} {vel_peak_idx:>8d} "
                    f"{lag_to_accel:>+7d} {lag_to_vel:>+7d} "
                    f"{na(corr_j_a):>8} {na(corr_j_v):>8} "
                    f"{nai(xcorr_a):>6} {nai(xcorr_v):>6}  "
                    f"{peak_sym:>6} {pk_z_s:>6} {pk_la_s:>8} {pk_lv_s:>8}  {timing}"
                )

        # Stim-level summary
        n_leads     = sum(1 for r in session_rows if r['timing'] == 'leads')
        n_lags      = sum(1 for r in session_rows if r['timing'] == 'lags')
        n_simul     = sum(1 for r in session_rows if r['timing'] == 'simultaneous')
        n_with_peak = sum(1 for r in session_rows if r['has_j_peak'])

        # Mean lag of the J peak (only sessions that actually have one)
        pk_lags_a = [r['j_peak_lag_to_accel'] for r in session_rows
                     if r['has_j_peak'] and not np.isnan(r['j_peak_lag_to_accel'])]
        pk_lags_v = [r['j_peak_lag_to_vel']   for r in session_rows
                     if r['has_j_peak'] and not np.isnan(r['j_peak_lag_to_vel'])]

        def _mn(lst): return np.mean(lst)    if lst else np.nan
        def _sd(lst): return np.std(lst)     if lst else np.nan
        def _nmn(lst): return np.nanmean(lst) if lst else np.nan

        results['by_stimulus'][stim] = {
            'n_sessions':              n_sess,
            'n_j_leads':               n_leads,
            'n_j_lags':                n_lags,
            'n_simultaneous':          n_simul,
            'n_with_j_peak':           n_with_peak,
            # J jump lags
            'mean_lag_accel':          _mn(stim_lag_accel),
            'std_lag_accel':           _sd(stim_lag_accel),
            'mean_lag_vel':            _mn(stim_lag_vel),
            'std_lag_vel':             _sd(stim_lag_vel),
            'mean_dist_accel':         _mn(stim_dist_accel),
            'mean_dist_vel':           _mn(stim_dist_vel),
            'mean_corr_j_accel':       _nmn(stim_corr_accel),
            'mean_corr_j_vel':         _nmn(stim_corr_vel),
            'mean_xcorr_accel':        _nmn(stim_xcorr_accel),
            'mean_xcorr_vel':          _nmn(stim_xcorr_vel),
            'mean_j_peak_lag_accel':   np.mean(pk_lags_a) if pk_lags_a else np.nan,
            'mean_j_peak_lag_vel':     np.mean(pk_lags_v) if pk_lags_v else np.nan,
            # Firing rate extrema lags
            'mean_fr_min_lag_accel':   _mn(stim_fr_min_lag_accel),
            'std_fr_min_lag_accel':    _sd(stim_fr_min_lag_accel),
            'mean_fr_min_lag_vel':     _mn(stim_fr_min_lag_vel),
            'std_fr_min_lag_vel':      _sd(stim_fr_min_lag_vel),
            'mean_fr_max_lag_accel':   _mn(stim_fr_max_lag_accel),
            'std_fr_max_lag_accel':    _sd(stim_fr_max_lag_accel),
            'mean_fr_max_lag_vel':     _mn(stim_fr_max_lag_vel),
            'std_fr_max_lag_vel':      _sd(stim_fr_max_lag_vel),
            # Energy extrema lags
            'mean_en_min_lag_accel':   _mn(stim_en_min_lag_accel),
            'std_en_min_lag_accel':    _sd(stim_en_min_lag_accel),
            'mean_en_min_lag_vel':     _mn(stim_en_min_lag_vel),
            'std_en_min_lag_vel':      _sd(stim_en_min_lag_vel),
            'mean_en_max_lag_accel':   _mn(stim_en_max_lag_accel),
            'std_en_max_lag_accel':    _sd(stim_en_max_lag_accel),
            'mean_en_max_lag_vel':     _mn(stim_en_max_lag_vel),
            'std_en_max_lag_vel':      _sd(stim_en_max_lag_vel),
            'sessions':                session_rows,
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
            print(f"  J PEAK (prom/bg90>{peak_threshold}, smooth={smooth_sigma}): "
                  f"{n_with_peak}/{n_sess} sessions have a large peak")
            if pk_lags_a:
                pk_a_dir = 'leads' if s['mean_j_peak_lag_accel'] < 0 else 'lags'
                pk_v_dir = 'leads' if s['mean_j_peak_lag_vel']   < 0 else 'lags'
                print(f"    Mean J-peak lag → accel: {s['mean_j_peak_lag_accel']:+.1f} bins  "
                      f"(J-peak {pk_a_dir})")
                print(f"    Mean J-peak lag → vel:   {s['mean_j_peak_lag_vel']:+.1f} bins  "
                      f"(J-peak {pk_v_dir})")

    n_with_peak_all = sum(all_has_peak)
    all_pk_lags_a   = [r['j_peak_lag_to_accel'] for r in results['all_sessions']
                       if r['has_j_peak'] and not np.isnan(r['j_peak_lag_to_accel'])]
    all_pk_lags_v   = [r['j_peak_lag_to_vel']   for r in results['all_sessions']
                       if r['has_j_peak'] and not np.isnan(r['j_peak_lag_to_vel'])]

    def _omn(lst): return np.mean(lst)    if lst else np.nan
    def _osd(lst): return np.std(lst)     if lst else np.nan
    def _onmn(lst): return np.nanmean(lst) if lst else np.nan

    results['overall'] = {
        'n_sessions':              total_sessions,
        'n_with_j_peak':           n_with_peak_all,
        # J jump lags
        'mean_lag_accel':          _omn(all_lag_accel),
        'std_lag_accel':           _osd(all_lag_accel),
        'mean_lag_vel':            _omn(all_lag_vel),
        'std_lag_vel':             _osd(all_lag_vel),
        'mean_dist_accel':         _omn(all_dist_accel),
        'mean_dist_vel':           _omn(all_dist_vel),
        'mean_corr_j_accel':       _onmn(all_corr_accel),
        'mean_corr_j_vel':         _onmn(all_corr_vel),
        'mean_xcorr_accel':        _onmn(all_xcorr_accel),
        'mean_xcorr_vel':          _onmn(all_xcorr_vel),
        'mean_j_peak_lag_accel':   np.mean(all_pk_lags_a) if all_pk_lags_a else np.nan,
        'mean_j_peak_lag_vel':     np.mean(all_pk_lags_v) if all_pk_lags_v else np.nan,
        # Firing rate extrema lags
        'mean_fr_min_lag_accel':   _omn(all_fr_min_lag_accel),
        'std_fr_min_lag_accel':    _osd(all_fr_min_lag_accel),
        'mean_fr_min_lag_vel':     _omn(all_fr_min_lag_vel),
        'std_fr_min_lag_vel':      _osd(all_fr_min_lag_vel),
        'mean_fr_max_lag_accel':   _omn(all_fr_max_lag_accel),
        'std_fr_max_lag_accel':    _osd(all_fr_max_lag_accel),
        'mean_fr_max_lag_vel':     _omn(all_fr_max_lag_vel),
        'std_fr_max_lag_vel':      _osd(all_fr_max_lag_vel),
        # Energy extrema lags
        'mean_en_min_lag_accel':   _omn(all_en_min_lag_accel),
        'std_en_min_lag_accel':    _osd(all_en_min_lag_accel),
        'mean_en_min_lag_vel':     _omn(all_en_min_lag_vel),
        'std_en_min_lag_vel':      _osd(all_en_min_lag_vel),
        'mean_en_max_lag_accel':   _omn(all_en_max_lag_accel),
        'std_en_max_lag_accel':    _osd(all_en_max_lag_accel),
        'mean_en_max_lag_vel':     _omn(all_en_max_lag_vel),
        'std_en_max_lag_vel':      _osd(all_en_max_lag_vel),
    }

    if verbose:
        ov = results['overall']
        n_leads_all = sum(1 for r in results['all_sessions'] if r['timing'] == 'leads')
        n_lags_all  = sum(1 for r in results['all_sessions'] if r['timing'] == 'lags')
        n_simul_all = sum(1 for r in results['all_sessions'] if r['timing'] == 'simultaneous')
        lag_a_dir = 'leads' if ov['mean_lag_accel'] < 0 else 'lags'
        lag_v_dir = 'leads' if ov['mean_lag_vel']   < 0 else 'lags'
        print(f"\n{'═'*90}")
        print(f"  OVERALL  ({total_sessions} sessions)")
        print(f"{'═'*90}")
        print(f"  J leads: {n_leads_all}  |  J lags: {n_lags_all}  |  simultaneous: {n_simul_all}")
        print(f"  Mean lag → accel: {ov['mean_lag_accel']:+.1f} ± {ov['std_lag_accel']:.1f} bins  "
              f"(J {lag_a_dir})")
        print(f"  Mean lag → vel:   {ov['mean_lag_vel']:+.1f} ± {ov['std_lag_vel']:.1f} bins  "
              f"(J {lag_v_dir})")
        print(f"  Mean Corr(J, accel): {ov['mean_corr_j_accel']:.3f}"
              f"  |  XCorr lag: {ov['mean_xcorr_accel']:+.1f} bins")
        print(f"  Mean Corr(J, vel):   {ov['mean_corr_j_vel']:.3f}"
              f"  |  XCorr lag: {ov['mean_xcorr_vel']:+.1f} bins")
        print(f"\n  J PEAK (prom/bg90>{peak_threshold}, smooth={smooth_sigma}): "
              f"{n_with_peak_all}/{total_sessions} sessions have a large J peak in the window")
        if all_pk_lags_a:
            pk_a_dir = 'leads' if ov['mean_j_peak_lag_accel'] < 0 else 'lags'
            pk_v_dir = 'leads' if ov['mean_j_peak_lag_vel']   < 0 else 'lags'
            print(f"    Mean J-peak lag → accel: {ov['mean_j_peak_lag_accel']:+.1f} bins  "
                  f"(J-peak {pk_a_dir})")
            print(f"    Mean J-peak lag → vel:   {ov['mean_j_peak_lag_vel']:+.1f} bins  "
                  f"(J-peak {pk_v_dir})")

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
                                timing, save_dir,
                                has_j_peak=False, j_peak_idx=None,
                                j_peak_value=np.nan, j_peak_z=np.nan,
                                j_peak_lag_to_accel=np.nan,
                                j_peak_lag_to_vel=np.nan,
                                smooth_sigma=5.0,
                                fr_ts=None, energy_ts=None,
                                fr_min_idx=None, fr_max_idx=None,
                                fr_min_lag_accel=None, fr_min_lag_vel=None,
                                fr_max_lag_accel=None, fr_max_lag_vel=None,
                                en_min_idx=None, en_max_idx=None,
                                en_min_lag_accel=None, en_min_lag_vel=None,
                                en_max_lag_accel=None, en_max_lag_vel=None,
                                window=None,
                                has_fr_peak=False, fr_peak_idx=None,
                                fr_peak_z=np.nan, fr_peak_lag_accel=None,
                                fr_peak_lag_vel=None,
                                has_fr_trough=False, fr_trough_idx=None,
                                fr_trough_z=np.nan, fr_trough_lag_accel=None,
                                fr_trough_lag_vel=None,
                                has_en_peak=False, en_peak_idx=None,
                                en_peak_z=np.nan, en_peak_lag_accel=None,
                                en_peak_lag_vel=None,
                                has_en_trough=False, en_trough_idx=None,
                                en_trough_z=np.nan, en_trough_lag_accel=None,
                                en_trough_lag_vel=None):
    """5-panel figure: velocity | acceleration | firing rate | energy | J coupling."""

    na_fmt  = lambda v: f"{v:.3f}" if (isinstance(v, float) and not np.isnan(v)) else "n/a"
    nai_fmt = lambda v: f"{v:+d}"  if (isinstance(v, (int, np.integer))) else "n/a"
    lag_dir = timing

    panel_color = {'leads': 'limegreen', 'lags': 'tomato',
                   'simultaneous': 'gold'}[timing]

    peak_line = ""
    if has_j_peak and j_peak_idx is not None:
        peak_line = (f"\nJ PEAK: idx={j_peak_idx}  ratio={na_fmt(j_peak_z)}  "
                     f"lag→A={nai_fmt(j_peak_lag_to_accel)}  "
                     f"lag→V={nai_fmt(j_peak_lag_to_vel)}")
    else:
        peak_line = f"\nNo large J peak (ratio={na_fmt(j_peak_z)})"

    fr_det_line = ""
    if has_fr_peak:
        fr_det_line += (f"\nFR PEAK: idx={fr_peak_idx}  ratio={na_fmt(fr_peak_z)}  "
                        f"lag→A={fr_peak_lag_accel:+d}  lag→V={fr_peak_lag_vel:+d}"
                        if fr_peak_lag_accel is not None else f"\nFR PEAK: idx={fr_peak_idx}  ratio={na_fmt(fr_peak_z)}")
    if has_fr_trough:
        fr_det_line += (f"\nFR TROUGH: idx={fr_trough_idx}  ratio={na_fmt(fr_trough_z)}  "
                        f"lag→A={fr_trough_lag_accel:+d}  lag→V={fr_trough_lag_vel:+d}"
                        if fr_trough_lag_accel is not None else f"\nFR TROUGH: idx={fr_trough_idx}  ratio={na_fmt(fr_trough_z)}")
    en_det_line = ""
    if has_en_peak:
        en_det_line += (f"\nEN PEAK: idx={en_peak_idx}  ratio={na_fmt(en_peak_z)}  "
                        f"lag→A={en_peak_lag_accel:+d}  lag→V={en_peak_lag_vel:+d}"
                        if en_peak_lag_accel is not None else f"\nEN PEAK: idx={en_peak_idx}  ratio={na_fmt(en_peak_z)}")
    if has_en_trough:
        en_det_line += (f"\nEN TROUGH: idx={en_trough_idx}  ratio={na_fmt(en_trough_z)}  "
                        f"lag→A={en_trough_lag_accel:+d}  lag→V={en_trough_lag_vel:+d}"
                        if en_trough_lag_accel is not None else f"\nEN TROUGH: idx={en_trough_idx}  ratio={na_fmt(en_trough_z)}")

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"Session {session}  |  Stim {stim}  |  J {lag_dir.upper()} kinematics\n"
        f"J jump idx={j_jump_idx}  |ΔJ|={na_fmt(j_jump_mag)}\n"
        f"Accel peak={accel_peak_idx} (lag {lag_to_accel:+d})  "
        f"Vel peak={vel_peak_idx} (lag {lag_to_vel:+d})\n"
        f"Corr(J,accel)={na_fmt(corr_j_accel)}  XClag={xcorr_lag_accel:+.0f}  |  "
        f"Corr(J,vel)={na_fmt(corr_j_vel)}  XClag={xcorr_lag_vel:+.0f}"
        f"{peak_line}{fr_det_line}{en_det_line}",
        fontsize=8, fontweight='bold'
    )

    # Shared reference lines + window shading helper
    w_lo = window[0] if window is not None else None
    w_hi = window[1] if window is not None else None

    def _ref_vlines(ax, extra_idx=None, extra_label=None, extra_color='crimson'):
        # Highlight the search window as a shaded band
        if w_lo is not None and w_hi is not None:
            ax.axvspan(w_lo, w_hi, color='gold', alpha=0.15, zorder=0,
                       label=f'Search window [{w_lo}–{w_hi}]')
            ax.axvline(w_lo, color='goldenrod', linestyle='--', linewidth=1.0,
                       alpha=0.55, zorder=1)
            ax.axvline(w_hi, color='goldenrod', linestyle='--', linewidth=1.0,
                       alpha=0.55, zorder=1)
        ax.axvline(accel_peak_idx, color='green',       linestyle='--',
                   linewidth=1.5, alpha=0.6, label=f'Accel peak ({accel_peak_idx})')
        ax.axvline(vel_peak_idx,   color='deepskyblue', linestyle='--',
                   linewidth=1.5, alpha=0.6, label=f'Vel peak ({vel_peak_idx})')
        if extra_idx is not None:
            ax.axvline(extra_idx, color=extra_color, linestyle=':', linewidth=1.5,
                       alpha=0.8, label=extra_label)

    # ── Panel 1: Velocity ──────────────────────────────────────────────────
    axes[0].plot(vel_ts, color='navy', linewidth=1.5, label='X velocity')
    _ref_vlines(axes[0], j_jump_idx, f'J jump ({j_jump_idx})')
    axes[0].set_ylabel("X Velocity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc='upper left')

    # ── Panel 2: Acceleration ─────────────────────────────────────────────
    axes[1].plot(accel_ts, color='steelblue', linewidth=1.5, label='X acceleration')
    _ref_vlines(axes[1], j_jump_idx, f'J jump ({j_jump_idx})')
    axes[1].set_ylabel("X Acceleration")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc='upper left')

    # ── Panel 3: Firing Rate ───────────────────────────────────────────────
    ax_fr = axes[2]
    if fr_ts is not None and len(fr_ts) > 0:
        ax_fr.plot(fr_ts, color='darkgreen', linewidth=1.5, label='Firing rate')
        # Raw window min/max (grey, background reference)
        if fr_min_idx is not None:
            ax_fr.axvline(fr_min_idx, color='silver', linestyle=':', linewidth=1.2, alpha=0.6,
                          label=f'FR window min ({fr_min_idx})')
        if fr_max_idx is not None:
            ax_fr.axvline(fr_max_idx, color='gray',   linestyle=':', linewidth=1.2, alpha=0.6,
                          label=f'FR window max ({fr_max_idx})')
        # Detected peak (★ orange)
        if has_fr_peak and fr_peak_idx is not None:
            ax_fr.axvline(fr_peak_idx, color='darkorange', linestyle='-.', linewidth=2.2,
                          label=f'FR PEAK ({fr_peak_idx})  ratio={na_fmt(fr_peak_z)}')
            if fr_peak_idx < len(fr_ts):
                ax_fr.scatter([fr_peak_idx], [fr_ts[fr_peak_idx]], color='darkorange',
                              zorder=6, s=90, marker='*')
        # Detected trough (★ purple)
        if has_fr_trough and fr_trough_idx is not None:
            ax_fr.axvline(fr_trough_idx, color='mediumpurple', linestyle='-.', linewidth=2.2,
                          label=f'FR TROUGH ({fr_trough_idx})  ratio={na_fmt(fr_trough_z)}')
            if fr_trough_idx < len(fr_ts):
                ax_fr.scatter([fr_trough_idx], [fr_ts[fr_trough_idx]], color='mediumpurple',
                              zorder=6, s=90, marker='*')
        _ref_vlines(ax_fr, j_jump_idx, f'J jump ({j_jump_idx})')
    else:
        ax_fr.text(0.5, 0.5, 'No firing rate data', transform=ax_fr.transAxes,
                   ha='center', va='center', color='gray')
    ax_fr.set_ylabel("Firing Rate")
    ax_fr.grid(alpha=0.3)
    ax_fr.legend(fontsize=7, loc='upper left')

    # ── Panel 4: Energy ────────────────────────────────────────────────────
    ax_en = axes[3]
    if energy_ts is not None and len(energy_ts) > 0:
        ax_en.plot(energy_ts, color='saddlebrown', linewidth=1.5, label='Energy')
        # Raw window min/max (grey, background reference)
        if en_min_idx is not None:
            ax_en.axvline(en_min_idx, color='silver', linestyle=':', linewidth=1.2, alpha=0.6,
                          label=f'En window min ({en_min_idx})')
        if en_max_idx is not None:
            ax_en.axvline(en_max_idx, color='gray',   linestyle=':', linewidth=1.2, alpha=0.6,
                          label=f'En window max ({en_max_idx})')
        # Detected peak (★ red-orange)
        if has_en_peak and en_peak_idx is not None:
            ax_en.axvline(en_peak_idx, color='tomato', linestyle='-.', linewidth=2.2,
                          label=f'EN PEAK ({en_peak_idx})  ratio={na_fmt(en_peak_z)}')
            if en_peak_idx < len(energy_ts):
                ax_en.scatter([en_peak_idx], [energy_ts[en_peak_idx]], color='tomato',
                              zorder=6, s=90, marker='*')
        # Detected trough (★ teal)
        if has_en_trough and en_trough_idx is not None:
            ax_en.axvline(en_trough_idx, color='teal', linestyle='-.', linewidth=2.2,
                          label=f'EN TROUGH ({en_trough_idx})  ratio={na_fmt(en_trough_z)}')
            if en_trough_idx < len(energy_ts):
                ax_en.scatter([en_trough_idx], [energy_ts[en_trough_idx]], color='teal',
                              zorder=6, s=90, marker='*')
        _ref_vlines(ax_en, j_jump_idx, f'J jump ({j_jump_idx})')
    else:
        ax_en.text(0.5, 0.5, 'No energy data', transform=ax_en.transAxes,
                   ha='center', va='center', color='gray')
    ax_en.set_ylabel("Energy")
    ax_en.grid(alpha=0.3)
    ax_en.legend(fontsize=7, loc='upper left')

    # ── Panel 5: J + dJ/dt with jump highlighted ──────────────────────────
    ax5 = axes[4]
    ax5_twin = ax5.twinx()

    ax5.plot(j_ts, color='darkorchid', linewidth=1.0, alpha=0.5, label='J (raw)')
    if len(j_ts) > 2:
        j_smooth_plot = _gaussian_smooth(j_ts, smooth_sigma)
        ax5.plot(j_smooth_plot, color='indigo', linewidth=2.0,
                 label=f'J smoothed (σ={smooth_sigma:.0f})')
    if w_lo is not None and w_hi is not None:
        ax5.axvspan(w_lo, w_hi, color='gold', alpha=0.15, zorder=0,
                    label=f'Search window [{w_lo}–{w_hi}]')
        ax5.axvline(w_lo, color='goldenrod', linestyle='--', linewidth=1.0, alpha=0.55, zorder=1)
        ax5.axvline(w_hi, color='goldenrod', linestyle='--', linewidth=1.0, alpha=0.55, zorder=1)
    ax5.axvline(j_jump_idx, color='crimson', linewidth=2.5,
                label=f'J jump (|ΔJ|={na_fmt(j_jump_mag)})')
    ax5.axvline(accel_peak_idx, color='green',       linestyle='--', linewidth=1.5, alpha=0.6,
                label='Accel peak')
    ax5.axvline(vel_peak_idx,   color='deepskyblue', linestyle='--', linewidth=1.5, alpha=0.6,
                label='Vel peak')

    if has_j_peak and j_peak_idx is not None:
        ax5.axvline(j_peak_idx, color='darkorange', linewidth=2.0, linestyle='-.',
                    label=f'J PEAK (ratio={na_fmt(j_peak_z)}, lag→A={nai_fmt(j_peak_lag_to_accel)})')
        if not np.isnan(j_peak_value):
            ax5.scatter([j_peak_idx], [j_peak_value], color='darkorange',
                        zorder=5, s=80, marker='*')

    ax5.set_facecolor((*mcolors.to_rgb(panel_color), 0.10))
    ax5.set_ylabel("J (coupling)", color='darkorchid')
    ax5.tick_params(axis='y', labelcolor='darkorchid')

    if len(j_ts) > 1:
        dj = np.abs(np.diff(j_ts, prepend=j_ts[0]))
        ax5_twin.plot(dj, color='salmon', linewidth=1.0, alpha=0.6, linestyle='-',
                      label='|dJ/dt|')
        ax5_twin.set_ylabel("|dJ/dt|", color='salmon')
        ax5_twin.tick_params(axis='y', labelcolor='salmon')

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    ax5.set_xlabel("Time bin")
    ax5.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"stim{stim}_{session}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator: analysis + plots + CSV
# ---------------------------------------------------------------------------

def j_kinematics_with_plots(stim_sessions_extrema, window, output_dir,
                             reference="acceleration", peak_threshold=1.5,
                             smooth_sigma=5.0, verbose=True):
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
        stim_sessions_extrema, window=window, reference=reference,
        peak_threshold=peak_threshold, smooth_sigma=smooth_sigma, verbose=verbose
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

        accel_ts  = _acceleration_timeseries(od)
        vel_ts    = _velocity_timeseries(od)
        j_ts      = _mean_timeseries(od, 'j')
        fr_ts     = _mean_timeseries(od, 'firing_rate') if 'firing_rate' in od.columns else np.array([])
        energy_ts = _mean_timeseries(od, 'energy')      if 'energy'       in od.columns else np.array([])

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
            has_j_peak=row['has_j_peak'],
            j_peak_idx=row['j_peak_idx'],
            j_peak_value=row['j_peak_value'],
            j_peak_z=row['j_peak_z'],
            j_peak_lag_to_accel=row['j_peak_lag_to_accel'],
            j_peak_lag_to_vel=row['j_peak_lag_to_vel'],
            smooth_sigma=smooth_sigma,
            fr_ts=fr_ts,     energy_ts=energy_ts,
            fr_min_idx=row['fr_min_idx'],   fr_max_idx=row['fr_max_idx'],
            fr_min_lag_accel=row['fr_min_lag_accel'], fr_min_lag_vel=row['fr_min_lag_vel'],
            fr_max_lag_accel=row['fr_max_lag_accel'], fr_max_lag_vel=row['fr_max_lag_vel'],
            en_min_idx=row['en_min_idx'],   en_max_idx=row['en_max_idx'],
            en_min_lag_accel=row['en_min_lag_accel'], en_min_lag_vel=row['en_min_lag_vel'],
            en_max_lag_accel=row['en_max_lag_accel'], en_max_lag_vel=row['en_max_lag_vel'],
            window=window,
            has_fr_peak=row['has_fr_peak'],     fr_peak_idx=row['fr_peak_idx'],
            fr_peak_z=row['fr_peak_z'],
            fr_peak_lag_accel=row['fr_peak_lag_accel'] if not (isinstance(row['fr_peak_lag_accel'], float) and np.isnan(row['fr_peak_lag_accel'])) else None,
            fr_peak_lag_vel=row['fr_peak_lag_vel']     if not (isinstance(row['fr_peak_lag_vel'],   float) and np.isnan(row['fr_peak_lag_vel']))   else None,
            has_fr_trough=row['has_fr_trough'],   fr_trough_idx=row['fr_trough_idx'],
            fr_trough_z=row['fr_trough_z'],
            fr_trough_lag_accel=row['fr_trough_lag_accel'] if not (isinstance(row['fr_trough_lag_accel'], float) and np.isnan(row['fr_trough_lag_accel'])) else None,
            fr_trough_lag_vel=row['fr_trough_lag_vel']     if not (isinstance(row['fr_trough_lag_vel'],   float) and np.isnan(row['fr_trough_lag_vel']))   else None,
            has_en_peak=row['has_en_peak'],     en_peak_idx=row['en_peak_idx'],
            en_peak_z=row['en_peak_z'],
            en_peak_lag_accel=row['en_peak_lag_accel'] if not (isinstance(row['en_peak_lag_accel'], float) and np.isnan(row['en_peak_lag_accel'])) else None,
            en_peak_lag_vel=row['en_peak_lag_vel']     if not (isinstance(row['en_peak_lag_vel'],   float) and np.isnan(row['en_peak_lag_vel']))   else None,
            has_en_trough=row['has_en_trough'],   en_trough_idx=row['en_trough_idx'],
            en_trough_z=row['en_trough_z'],
            en_trough_lag_accel=row['en_trough_lag_accel'] if not (isinstance(row['en_trough_lag_accel'], float) and np.isnan(row['en_trough_lag_accel'])) else None,
            en_trough_lag_vel=row['en_trough_lag_vel']     if not (isinstance(row['en_trough_lag_vel'],   float) and np.isnan(row['en_trough_lag_vel']))   else None,
        )

    # Session-level CSV (includes FR, Energy, and J fields)
    summary_df = pd.DataFrame([{
        'stimulus':             r['stimulus'],
        'session':              r['session'],
        'accel_peak_idx':       r['accel_peak_idx'],
        'vel_peak_idx':         r['vel_peak_idx'],
        # Firing rate
        'fr_min_idx':           r['fr_min_idx'],
        'fr_max_idx':           r['fr_max_idx'],
        'fr_min_lag_accel':     r['fr_min_lag_accel'],
        'fr_min_lag_vel':       r['fr_min_lag_vel'],
        'fr_max_lag_accel':     r['fr_max_lag_accel'],
        'fr_max_lag_vel':       r['fr_max_lag_vel'],
        'fr_min_closer':        r['fr_min_closer'],
        'fr_max_closer':        r['fr_max_closer'],
        # Energy
        'en_min_idx':           r['en_min_idx'],
        'en_max_idx':           r['en_max_idx'],
        'en_min_lag_accel':     r['en_min_lag_accel'],
        'en_min_lag_vel':       r['en_min_lag_vel'],
        'en_max_lag_accel':     r['en_max_lag_accel'],
        'en_max_lag_vel':       r['en_max_lag_vel'],
        'en_min_closer':        r['en_min_closer'],
        'en_max_closer':        r['en_max_closer'],
        # J jump
        'j_jump_idx':           r['j_jump_idx'],
        'j_jump_mag':           r['j_jump_mag'],
        'lag_to_accel':         r['lag_to_accel'],
        'lag_to_vel':           r['lag_to_vel'],
        'dist_to_accel':        r['dist_to_accel'],
        'dist_to_vel':          r['dist_to_vel'],
        'corr_j_accel':         r['corr_j_accel'],
        'corr_j_vel':           r['corr_j_vel'],
        'xcorr_lag_accel':      r['xcorr_lag_accel'],
        'xcorr_lag_vel':        r['xcorr_lag_vel'],
        'timing':               r['timing'],
        # J peak
        'has_j_peak':           r['has_j_peak'],
        'j_peak_idx':           r['j_peak_idx'],
        'j_peak_value':         r['j_peak_value'],
        'j_peak_z':             r['j_peak_z'],
        'j_peak_lag_to_accel':  r['j_peak_lag_to_accel'],
        'j_peak_lag_to_vel':    r['j_peak_lag_to_vel'],
        # FR detected peak/trough
        'has_fr_peak':          r['has_fr_peak'],
        'fr_peak_idx':          r['fr_peak_idx'],
        'fr_peak_z':            r['fr_peak_z'],
        'fr_peak_lag_accel':    r['fr_peak_lag_accel'],
        'fr_peak_lag_vel':      r['fr_peak_lag_vel'],
        'has_fr_trough':        r['has_fr_trough'],
        'fr_trough_idx':        r['fr_trough_idx'],
        'fr_trough_z':          r['fr_trough_z'],
        'fr_trough_lag_accel':  r['fr_trough_lag_accel'],
        'fr_trough_lag_vel':    r['fr_trough_lag_vel'],
        # Energy detected peak/trough
        'has_en_peak':          r['has_en_peak'],
        'en_peak_idx':          r['en_peak_idx'],
        'en_peak_z':            r['en_peak_z'],
        'en_peak_lag_accel':    r['en_peak_lag_accel'],
        'en_peak_lag_vel':      r['en_peak_lag_vel'],
        'has_en_trough':        r['has_en_trough'],
        'en_trough_idx':        r['en_trough_idx'],
        'en_trough_z':          r['en_trough_z'],
        'en_trough_lag_accel':  r['en_trough_lag_accel'],
        'en_trough_lag_vel':    r['en_trough_lag_vel'],
    } for r in results['all_sessions']])
    summary_df.to_csv(os.path.join(output_dir, 'session_summary_j_kinematics.csv'), index=False)

    n_leads = (summary_df['timing'] == 'leads').sum()
    n_lags  = (summary_df['timing'] == 'lags').sum()
    n_sim   = (summary_df['timing'] == 'simultaneous').sum()

    # Box-and-whisker arbitration plot
    plot_arbitration_boxwhisker(results['all_sessions'], output_dir, window=window)

    if verbose:
        print(f"\n  Plots saved:")
        print(f"    {n_leads} J-leads sessions  → {leads_dir}")
        print(f"    {n_lags}  J-lags sessions   → {lags_dir}")
        print(f"    {n_sim}  simultaneous      → {simul_dir}")

    results['summary_df'] = summary_df
    return results


# ---------------------------------------------------------------------------
# Box-and-whisker arbitration plot
# ---------------------------------------------------------------------------

def plot_arbitration_boxwhisker(all_sessions, output_dir, window=None):
    """
    Box-and-whisker plot showing the lag (in bins) of each signal's extremum
    relative to velocity and acceleration peaks.

    Six signal groups on the x-axis:
        FR min | FR max | Energy min | Energy max | J jump | J peak

    Each group has two boxes:
        Blue  = lag relative to acceleration peak
        Orange = lag relative to velocity peak

    A horizontal dashed line at y=0 marks coincidence.
    Negative lag → signal *leads* the kinematic event.
    Positive lag → signal *lags* the kinematic event.
    """
    # ── Collect lag arrays ────────────────────────────────────────────────
    signals = [
        ('FR\nmin',    'fr_min_lag_accel',   'fr_min_lag_vel'),
        ('FR\nmax',    'fr_max_lag_accel',   'fr_max_lag_vel'),
        ('Energy\nmin','en_min_lag_accel',   'en_min_lag_vel'),
        ('Energy\nmax','en_max_lag_accel',   'en_max_lag_vel'),
        ('J pairwise\njump',    'lag_to_accel',       'lag_to_vel'),
        ('J pairwise\npeak',    'j_peak_lag_to_accel','j_peak_lag_to_vel'),
    ]

    accel_data, vel_data, labels = [], [], []
    for label, accel_key, vel_key in signals:
        a_lags = [r[accel_key] for r in all_sessions
                  if accel_key in r and not (isinstance(r[accel_key], float) and np.isnan(r[accel_key]))]
        v_lags = [r[vel_key]   for r in all_sessions
                  if vel_key   in r and not (isinstance(r[vel_key],   float) and np.isnan(r[vel_key]))]
        accel_data.append(a_lags)
        vel_data.append(v_lags)
        labels.append(label)

    n_groups = len(labels)
    x = np.arange(n_groups)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    def _bplot(ax, positions, data, color, label_str, flier_props):
        bps = ax.boxplot(
            data, positions=positions, widths=width * 0.9,
            patch_artist=True, notch=False,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=flier_props,
            manage_ticks=False,
        )
        bps['boxes'][0].set_label(label_str)
        return bps

    fp_a = dict(marker='o', color='steelblue',   markersize=4, alpha=0.5, linestyle='none')
    fp_v = dict(marker='o', color='darkorange',  markersize=4, alpha=0.5, linestyle='none')

    for i, (a_dat, v_dat) in enumerate(zip(accel_data, vel_data)):
        if a_dat:
            _bplot(ax, [x[i] - width / 2], [a_dat], 'steelblue',
                   'vs Acceleration' if i == 0 else '_nolegend_', fp_a)
        if v_dat:
            _bplot(ax, [x[i] + width / 2], [v_dat], 'darkorange',
                   'vs Velocity' if i == 0 else '_nolegend_', fp_v)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='Zero lag (coincident)')

    # Compute y range for annotation placement
    all_flat = [v for lst in accel_data + vel_data for v in lst]
    y_top = (max(all_flat) * 1.08) if all_flat else 1.0

    # Annotate each group with the winner ("closer to" vote)
    for i, (label, accel_key, vel_key) in enumerate(signals):
        a_lags = accel_data[i]
        v_lags = vel_data[i]
        if a_lags and v_lags:
            med_a = np.median(np.abs(a_lags))
            med_v = np.median(np.abs(v_lags))
            winner = 'A' if med_a <= med_v else 'V'
            color  = 'steelblue' if winner == 'A' else 'darkorange'
            ax.text(x[i], y_top, f'→{winner}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Signal / Extremum", fontsize=12)
    ax.set_ylabel("Lag relative to kinematic peak (bins)\n(−  leads  |  +  lags)", fontsize=11)

    n_sess = len(all_sessions)
    win_str = f"window [{window[0]}–{window[1]}]" if window is not None else "window: n/a"
    ax.set_title(
        f"Arbitration: Signal extrema timing relative to Acceleration vs Velocity\n"
        f"(Blue = vs Accel, Orange = vs Vel  |  →A / →V = median closer to Accel / Velocity)\n"
        f"n = {n_sess} sessions  |  search {win_str}",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'arbitration_boxwhisker.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Arbitration box-whisker plot saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_arbitration_text_report(all_sessions, overall, by_stimulus, output_dir, args):
    """
    Write a plain-text summary of the arbitration results.

    Includes:
    - Per-session lead/lag table for FR (min/max), Energy (min/max), J jump, J peak
    - Per-stimulus summary: mean ± std and closer-to vote for each signal
    - Overall summary across all sessions
    """
    out_path = os.path.join(output_dir, 'arbitration_results.txt')

    def _fmt(v, fmt='+.1f'):
        if v is None:
            return '   n/a'
        try:
            if np.isnan(float(v)):
                return '   n/a'
        except (TypeError, ValueError):
            pass
        return format(v, fmt)

    def _closer(mean_dist_a, mean_dist_v):
        if np.isnan(mean_dist_a) or np.isnan(mean_dist_v):
            return 'n/a'
        return 'ACCEL' if mean_dist_a <= mean_dist_v else 'VEL'

    lines = []
    lines.append("=" * 110)
    lines.append("  ARBITRATION RESULTS: Firing Rate, Energy, and J vs Acceleration & Velocity")
    lines.append(f"  Generated: {date.today().isoformat()}")
    lines.append(f"  Data folder: {getattr(args, 'data_folder', 'N/A')}")
    lines.append(f"  Window: {getattr(args, 'window', 'N/A')}  |  "
                 f"Reps: {getattr(args, 'rep_start', 'N/A')}–{getattr(args, 'rep_end_exclusive', 'N/A') - 1 if hasattr(args, 'rep_end_exclusive') else 'N/A'}  |  "
                 f"Stimuli: {getattr(args, 'stim_min', 'N/A')}–{getattr(args, 'stim_max_exclusive', 'N/A') - 1 if hasattr(args, 'stim_max_exclusive') else 'N/A'}")
    lines.append("")
    lines.append("  LAG CONVENTION:")
    lines.append("    lag = signal_extremum_index − kinematic_peak_index")
    lines.append("    Negative → signal LEADS  kinematic event")
    lines.append("    Positive → signal LAGS   kinematic event")
    lines.append("    Zero     → coincident")
    lines.append("=" * 110)
    lines.append("")

    # ── Per-session table ─────────────────────────────────────────────────
    lines.append("─" * 110)
    lines.append("  PER-SESSION DETAIL")
    lines.append("─" * 110)
    hdr = (f"  {'Session':<12} {'Stim':>4}  "
           f"{'FR_min→A':>9} {'FR_min→V':>9} {'FR_max→A':>9} {'FR_max→V':>9}  "
           f"{'En_min→A':>9} {'En_min→V':>9} {'En_max→A':>9} {'En_max→V':>9}  "
           f"{'J_jump→A':>9} {'J_jump→V':>9}  "
           f"{'J_pk→A':>8} {'J_pk→V':>8}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for r in all_sessions:
        sess = str(r['session'])
        stim = str(r['stimulus'])
        jpa  = _fmt(r.get('j_peak_lag_to_accel', float('nan')))
        jpv  = _fmt(r.get('j_peak_lag_to_vel',   float('nan')))
        lines.append(
            f"  {sess:<12} {stim:>4}  "
            f"{_fmt(r['fr_min_lag_accel']):>9} {_fmt(r['fr_min_lag_vel']):>9} "
            f"{_fmt(r['fr_max_lag_accel']):>9} {_fmt(r['fr_max_lag_vel']):>9}  "
            f"{_fmt(r['en_min_lag_accel']):>9} {_fmt(r['en_min_lag_vel']):>9} "
            f"{_fmt(r['en_max_lag_accel']):>9} {_fmt(r['en_max_lag_vel']):>9}  "
            f"{_fmt(r['lag_to_accel']):>9} {_fmt(r['lag_to_vel']):>9}  "
            f"{jpa:>8} {jpv:>8}"
        )

    lines.append("")

    # ── Per-stimulus summary ──────────────────────────────────────────────
    lines.append("─" * 110)
    lines.append("  PER-STIMULUS SUMMARY  (mean ± std of lag, bins)  [→A closer to Accel | →V closer to Vel]")
    lines.append("─" * 110)

    signal_keys = [
        ('FR min',    'mean_fr_min_lag_accel', 'std_fr_min_lag_accel',
                      'mean_fr_min_lag_vel',   'std_fr_min_lag_vel'),
        ('FR max',    'mean_fr_max_lag_accel', 'std_fr_max_lag_accel',
                      'mean_fr_max_lag_vel',   'std_fr_max_lag_vel'),
        ('Energy min','mean_en_min_lag_accel', 'std_en_min_lag_accel',
                      'mean_en_min_lag_vel',   'std_en_min_lag_vel'),
        ('Energy max','mean_en_max_lag_accel', 'std_en_max_lag_accel',
                      'mean_en_max_lag_vel',   'std_en_max_lag_vel'),
        ('J jump',    'mean_lag_accel',        'std_lag_accel',
                      'mean_lag_vel',          'std_lag_vel'),
        ('J peak',    'mean_j_peak_lag_accel', None,
                      'mean_j_peak_lag_vel',   None),
    ]

    for stim_id in sorted(by_stimulus):
        s = by_stimulus[stim_id]
        lines.append(f"\n  Stimulus {stim_id}  ({s['n_sessions']} sessions)")
        lines.append(f"  {'Signal':<14} {'Mean→Accel':>12} {'Std→Accel':>10}  "
                     f"{'Mean→Vel':>10} {'Std→Vel':>10}   {'Closer to':>12}")
        lines.append("  " + "-" * 72)
        for sig_name, mk_a, sk_a, mk_v, sk_v in signal_keys:
            m_a = s.get(mk_a, float('nan'))
            s_a = s.get(sk_a, float('nan')) if sk_a else float('nan')
            m_v = s.get(mk_v, float('nan'))
            s_v = s.get(sk_v, float('nan')) if sk_v else float('nan')
            winner = _closer(abs(m_a) if not np.isnan(m_a) else float('nan'),
                             abs(m_v) if not np.isnan(m_v) else float('nan'))
            lines.append(
                f"  {sig_name:<14} {_fmt(m_a):>12} {_fmt(s_a, '.1f'):>10}  "
                f"{_fmt(m_v):>10} {_fmt(s_v, '.1f'):>10}   {winner:>12}"
            )

    lines.append("")

    # ── Overall summary ───────────────────────────────────────────────────
    lines.append("─" * 110)
    lines.append(f"  OVERALL SUMMARY  ({overall['n_sessions']} sessions total)")
    lines.append("─" * 110)
    lines.append(f"  {'Signal':<14} {'Mean→Accel':>12} {'Std→Accel':>10}  "
                 f"{'Mean→Vel':>10} {'Std→Vel':>10}   {'Closer to':>12}")
    lines.append("  " + "-" * 72)
    for sig_name, mk_a, sk_a, mk_v, sk_v in signal_keys:
        m_a = overall.get(mk_a, float('nan'))
        s_a = overall.get(sk_a, float('nan')) if sk_a else float('nan')
        m_v = overall.get(mk_v, float('nan'))
        s_v = overall.get(sk_v, float('nan')) if sk_v else float('nan')
        winner = _closer(abs(m_a) if not np.isnan(m_a) else float('nan'),
                         abs(m_v) if not np.isnan(m_v) else float('nan'))
        lines.append(
            f"  {sig_name:<14} {_fmt(m_a):>12} {_fmt(s_a, '.1f'):>10}  "
            f"{_fmt(m_v):>10} {_fmt(s_v, '.1f'):>10}   {winner:>12}"
        )

    # Overall "closer to" vote count (per-session)
    lines.append("")
    lines.append("  Per-session closer-to vote counts (across all sessions):")
    for label, accel_key, vel_key in [
        ('FR min',     'fr_min_closer', None),
        ('FR max',     'fr_max_closer', None),
        ('Energy min', 'en_min_closer', None),
        ('Energy max', 'en_max_closer', None),
    ]:
        n_a = sum(1 for r in all_sessions if r.get(accel_key) == 'accel')
        n_v = sum(1 for r in all_sessions if r.get(accel_key) == 'vel')
        total = n_a + n_v
        lines.append(f"    {label:<14}: {n_a}/{total} closer to ACCEL  |  {n_v}/{total} closer to VEL")

    n_j_a = sum(1 for r in all_sessions if r.get('dist_to_accel', float('inf')) <= r.get('dist_to_vel', float('inf')))
    n_j_v = len(all_sessions) - n_j_a
    lines.append(f"    {'J jump':<14}: {n_j_a}/{len(all_sessions)} closer to ACCEL  |  {n_j_v}/{len(all_sessions)} closer to VEL")

    jpk_a = sum(1 for r in all_sessions if r.get('has_j_peak') and
                not np.isnan(r.get('j_peak_lag_to_accel', float('nan'))) and
                not np.isnan(r.get('j_peak_lag_to_vel',   float('nan'))) and
                abs(r['j_peak_lag_to_accel']) <= abs(r['j_peak_lag_to_vel']))
    jpk_v = sum(1 for r in all_sessions if r.get('has_j_peak') and
                not np.isnan(r.get('j_peak_lag_to_accel', float('nan'))) and
                not np.isnan(r.get('j_peak_lag_to_vel',   float('nan'))) and
                abs(r['j_peak_lag_to_accel']) > abs(r['j_peak_lag_to_vel']))
    jpk_tot = jpk_a + jpk_v
    lines.append(f"    {'J peak':<14}: {jpk_a}/{jpk_tot} closer to ACCEL  |  {jpk_v}/{jpk_tot} closer to VEL"
                 f"  (sessions with detected peak)")

    lines.append("")
    lines.append("=" * 110)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Arbitration text report saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def _aggregate_overall(overall_list):
    """Merge a list of per-rep overall dicts by pooling numeric arrays."""
    if not overall_list:
        return {}
    keys = [k for k in overall_list[0] if k != 'n_sessions']
    agg = {'n_sessions': sum(o.get('n_sessions', 0) for o in overall_list)}
    for k in keys:
        vals = [o[k] for o in overall_list if k in o and not (isinstance(o[k], float) and np.isnan(o[k]))]
        agg[k] = float(np.mean(vals)) if vals else float('nan')
    return agg


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
        "Lag→A | Lag→V | r(J,A) | r(J,V) | Timing | J Peak? | Prom/bg90 | PkLag→A | PkLag→V |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rep, res in sorted(all_results.items()):
        for row in res['all_sessions']:
            mag   = f"{row['j_jump_mag']:.4f}"        if not np.isnan(row['j_jump_mag'])        else "n/a"
            rja   = f"{row['corr_j_accel']:.3f}"      if not np.isnan(row['corr_j_accel'])      else "n/a"
            rjv   = f"{row['corr_j_vel']:.3f}"        if not np.isnan(row['corr_j_vel'])        else "n/a"
            pk_z  = f"{row['j_peak_z']:.2f}"          if not np.isnan(row['j_peak_z'])          else "n/a"
            pk_la = f"{row['j_peak_lag_to_accel']:+d}" if (row['has_j_peak'] and
                        not np.isnan(row['j_peak_lag_to_accel'])) else "—"
            pk_lv = f"{row['j_peak_lag_to_vel']:+d}"   if (row['has_j_peak'] and
                        not np.isnan(row['j_peak_lag_to_vel']))   else "—"
            peak_yn = "**YES**" if row['has_j_peak'] else "no"
            lines.append(
                f"| {rep} | {row['stimulus']} | {row['session']} | "
                f"{row['j_jump_idx']} | {mag} | "
                f"{row['accel_peak_idx']} | {row['vel_peak_idx']} | "
                f"{row['lag_to_accel']:+d} | {row['lag_to_vel']:+d} | "
                f"{rja} | {rjv} | **{row['timing']}** | "
                f"{peak_yn} | {pk_z} | {pk_la} | {pk_lv} |"
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
                   default=[375, 450],
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
    p.add_argument("--peak_threshold", type=float, default=2.0,
                   help="Ratio threshold for declaring a J peak 'large': "
                        "window_prominence / bg_90th_percentile > threshold "
                        "(default 2.0 — window peak must be 2x more prominent "
                        "than typical background undulations)")
    p.add_argument("--smooth_sigma", type=float, default=5.0,
                   help="Gaussian smoothing width in time-bins applied to J "
                        "before peak detection (default 5.0). Larger values "
                        "require a broader, smoother bump to count as a peak.")
    p.add_argument("--sessions", type=str, nargs="+", default=None,
                   metavar="SESSION_ID",
                   help="Optional whitelist of session IDs to process (e.g. "
                        "--sessions 210425 210511 220515). The ID is matched "
                        "against the 6-digit session identifier extracted from "
                        "each per_reach_state.csv path. If omitted, all "
                        "sessions found under --data_folder are used.")
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

    stim_range       = range(args.stim_min, args.stim_max_exclusive)
    window           = [w_lo, w_hi]
    all_results      = {}
    session_whitelist = set(args.sessions) if args.sessions else None

    if session_whitelist:
        print(f"Session filter active — processing only: {sorted(session_whitelist)}")

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

            if session_whitelist and session_id not in session_whitelist:
                continue

            print(session)
            print(f"Session_id: {session_id}")
            df = pd.read_csv(session)
            if session_id in session_data:
                session_data[session_id] = pd.concat(
                    [session_data[session_id], df], ignore_index=True)
            else:
                session_data[session_id] = df

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
            peak_threshold=args.peak_threshold,
            smooth_sigma=args.smooth_sigma,
            verbose=not args.quiet,
        )
        all_results[rep] = results

        # Text arbitration report (per-rep)
        write_arbitration_text_report(
            results['all_sessions'],
            results['overall'],
            results['by_stimulus'],
            rep_out,
            args,
        )

        df_out = pd.concat(
            [extract_stats_to_df(results['by_stimulus'][stim], stim=stim, rep=rep)
             for stim in results['by_stimulus']],
            ignore_index=True,
        )
        df_out.to_csv(os.path.join(rep_out, "results_j_kinematics.csv"), index=False)
        print(f"Wrote {os.path.join(rep_out, 'results_j_kinematics.csv')}")

    # Across-rep text report (aggregated)
    if all_results:
        all_sess_combined = [r for res in all_results.values()
                             for r in res['all_sessions']]
        # Build aggregated overall/by_stimulus from first available rep or combine
        combined_overall = _aggregate_overall([res['overall'] for res in all_results.values()])
        combined_by_stim = {}
        for res in all_results.values():
            for stim, sdata in res['by_stimulus'].items():
                combined_by_stim.setdefault(stim, sdata)  # use first rep per stim

        write_arbitration_text_report(
            all_sess_combined,
            combined_overall,
            combined_by_stim,
            out_base,
            args,
        )

    if args.report_dir and all_results:
        write_j_kinematics_markdown(all_results, out_base, args.report_dir, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
