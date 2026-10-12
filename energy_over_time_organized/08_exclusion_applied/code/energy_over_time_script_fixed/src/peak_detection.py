"""
peak_detection.py
-----------------
Utilities for detecting significant peaks, troughs, and level-shifts in
1-D time series within a specified search window.

Public API
----------
gaussian_smooth(arr, sigma)
    Convolve a signal with a Gaussian kernel.

local_prominence(signal, idx, half)
    Local prominence of a candidate peak at `idx`.

detect_j_peak(j_ts, w_lo, w_hi, threshold_ratio, smooth_sigma)
    Detect a significant event in a J-coupling time series (bump OR step).

detect_signal_extremum(ts, w_lo, w_hi, threshold_ratio, smooth_sigma, kind)
    Generalised peak/trough detector for any signal.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def gaussian_smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve *arr* with a Gaussian kernel of given *sigma* (in samples)."""
    if sigma <= 0 or len(arr) < 3:
        return arr.astype(float)
    r      = int(np.ceil(3 * sigma))
    x      = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr.astype(float), kernel, mode='same')


def local_prominence(signal: np.ndarray, idx: int, half: int) -> float:
    """
    Prominence of a candidate peak at *idx* in *signal*:
        peak_value − max(left_valley_min, right_valley_min)
    where the valleys are searched within ±*half* bins of the peak.
    """
    left_seg  = signal[max(0, idx - half) : idx]
    right_seg = signal[idx + 1 : min(len(signal), idx + half + 1)]
    left_val  = float(left_seg.min())  if len(left_seg)  else float(signal[idx])
    right_val = float(right_seg.min()) if len(right_seg) else float(signal[idx])
    return float(signal[idx]) - max(left_val, right_val)


# ---------------------------------------------------------------------------
# Internal shared detection core
# ---------------------------------------------------------------------------

def _detect_peak_core(ts_work: np.ndarray, orig_ts: np.ndarray,
                      w_lo: int, w_hi: int,
                      threshold_ratio: float, smooth_sigma: float,
                      min_abs_z: float = 0.5,
                      min_signal_range: float = 0.0):
    """
    Shared detection logic used by both detect_j_peak and
    detect_signal_extremum.  Always searches for a *maximum* in ts_work
    (callers flip the signal for trough detection).

    Parameters
    ----------
    min_abs_z : float
        Absolute-amplitude gate (default 0.5).  After the prominence/level-
        shift criteria flag an event, the peak must also deviate from the
        pre-window baseline mean by at least *min_abs_z* × (full-signal std).
        This prevents tiny fluctuations on nearly-flat signals from passing
        purely on a relative-ratio score.
    min_signal_range : float
        Hard minimum on the full-signal (smoothed) peak-to-trough range
        (default 0.0 = disabled).  If the signal's entire dynamic range is
        below this value the timeseries is considered physically flat and no
        event is reported.  Use signal-specific units (e.g. 0.5 for J coupling
        which can span several units, but leave at 0.0 for normalised signals).

    Returns (has_event, event_idx, event_raw_value, ratio).
    """
    w_lo = max(0, int(w_lo))
    w_hi = min(len(ts_work), int(w_hi))
    if w_hi <= w_lo + 1 or len(ts_work) < 3:
        return False, w_lo, np.nan, np.nan

    smoothed = gaussian_smooth(ts_work, smooth_sigma)
    half     = max(10, (w_hi - w_lo) // 2)

    # Flat-signal guard: nothing can be exceptional if the range is negligible.
    signal_range = float(smoothed.max() - smoothed.min())
    if signal_range < 1e-6:
        return False, w_lo, float(orig_ts[w_lo]), 0.0

    # Absolute range gate: reject physically flat signals.
    if min_signal_range > 0 and signal_range < min_signal_range:
        return False, w_lo, float(orig_ts[w_lo]), 0.0

    # ── Candidate: tallest local maximum of the smoothed signal ──────────
    # Smoothing suppresses noise bumps, so the tallest local max is the
    # true peak — no prominence scoring needed for candidate selection
    # (prominence would penalise peaks near the window edge).
    win_local_maxima = [
        i for i in range(w_lo + 1, w_hi - 1)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    if win_local_maxima:
        event_idx = max(win_local_maxima, key=lambda i: smoothed[i])
    else:
        # Monotonic segment — fall back to global argmax within window.
        event_idx = w_lo + int(np.argmax(smoothed[w_lo:w_hi]))

    window_prom = local_prominence(smoothed, event_idx, half)

    # ── Criterion 1: Prominence (bump detection) ──────────────────────────
    bg_proms = []
    for i in range(1, len(smoothed) - 1):
        if w_lo <= i < w_hi:
            continue
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            bg_proms.append(local_prominence(smoothed, i, half))

    if bg_proms:
        bg_90    = float(np.percentile(bg_proms, 90))
        prom_c1  = (window_prom / bg_90) if bg_90 > 1e-10 else 0.0
    else:
        # No background peaks: normalise against full-signal std.
        # Use range/4 as a floor so residual noise on a flat signal never
        # inflates the ratio into a false positive.
        smooth_std = max(float(smoothed.std()), signal_range / 4.0, 1e-10)
        prom_c1    = window_prom / smooth_std

    # ── Criterion 2: Level shift (step / sustained excursion) ────────────
    bg_mask = np.ones(len(smoothed), dtype=bool)
    bg_mask[w_lo:w_hi] = False
    bg_vals = smoothed[bg_mask]
    level_z = 0.0
    if len(bg_vals) > 1:
        bg_mean = float(bg_vals.mean())
        bg_std  = float(bg_vals.std())
        # Same range/4 floor prevents floating-point noise inflation on a
        # flat background.
        bg_std   = max(bg_std, signal_range / 4.0, 1e-10)
        peak_val = float(smoothed[event_idx])
        win_min  = float(smoothed[w_lo:w_hi].min())
        level_z  = max((peak_val - bg_mean) / bg_std,
                       (bg_mean  - win_min)  / bg_std)

    # ── Combine ───────────────────────────────────────────────────────────
    ratio     = max(prom_c1, level_z)
    has_event = bool(prom_c1 > threshold_ratio or level_z > threshold_ratio)

    # ── Absolute amplitude gate ────────────────────────────────────────────
    # Even if the relative ratio passes, require the peak to be at least
    # min_abs_z standard deviations above the pre-window baseline mean.
    # This suppresses false positives on nearly-flat signals where tiny
    # fluctuations can inflate the ratio score.
    if has_event and min_abs_z > 0:
        full_std = float(smoothed.std())
        if full_std < 1e-10:
            has_event = False
        else:
            baseline = smoothed[:w_lo] if w_lo > 0 else smoothed[w_hi:]
            baseline_mean = float(baseline.mean()) if len(baseline) > 0 else float(smoothed.mean())
            abs_deviation = abs(float(smoothed[event_idx]) - baseline_mean) / full_std
            if abs_deviation < min_abs_z:
                has_event = False

    return has_event, event_idx, float(orig_ts[event_idx]), ratio


# ---------------------------------------------------------------------------
# Public detectors
# ---------------------------------------------------------------------------

def detect_j_peak(j_ts: np.ndarray, w_lo: int, w_hi: int,
                  threshold_ratio: float = 2.0,
                  smooth_sigma: float = 5.0,
                  min_abs_z: float = 0.5,
                  min_signal_range: float = 0.5):
    """
    Test whether the J-coupling signal has an exceptional event within the
    window [w_lo, w_hi].

    Two parallel criteria are tested; EITHER one is sufficient to flag the
    window as containing a meaningful event:

    Criterion 1 — Prominence (bump detection)
        Compares the smoothed-signal prominence of the window's tallest local
        maximum against the 90th-percentile prominence of all background local
        maxima outside the window.  Fires when the ratio exceeds
        *threshold_ratio*.  Works well for clear bump-shaped peaks.

    Criterion 2 — Level shift (step detection)
        Compares the window's maximum (or minimum) level to the background
        mean ± background std.  Fires when the z-score exceeds
        *threshold_ratio*.  Catches sustained plateaus or step-shifts that
        have no clear descent back to baseline.

    Absolute gate — min_abs_z
        After either criterion fires, the candidate peak must additionally
        deviate from the pre-window baseline mean by at least *min_abs_z*
        full-signal standard deviations.  This suppresses false positives on
        nearly-flat signals where tiny fluctuations inflate relative ratios.

    Absolute range gate — min_signal_range
        If the full J signal's peak-to-trough range (after smoothing) is below
        this value the signal is considered physically flat and no event is
        reported (default 0.5 J-coupling units).  Sessions where J barely
        moves at all — spanning < 0.5 units — cannot contain a meaningful peak.

    Parameters
    ----------
    j_ts             : full J time-series (numpy array)
    w_lo, w_hi       : search window boundaries (indices into j_ts)
    threshold_ratio  : detection threshold applied to both criteria (default 2.0)
    smooth_sigma     : Gaussian smoothing width in time-bins (default 5)
    min_abs_z        : absolute amplitude gate in units of full-signal std (default 0.5)
    min_signal_range : hard floor on the full-signal range in J units (default 0.5)

    Returns
    -------
    has_peak    : bool  — True if all criteria pass
    peak_idx    : int   — index of the candidate peak inside the window
    peak_value  : float — raw J value at peak_idx
    prom_ratio  : float — max(prominence_ratio, level_shift_z)
    """
    j_ts = np.asarray(j_ts, dtype=float)
    return _detect_peak_core(j_ts, j_ts, w_lo, w_hi, threshold_ratio, smooth_sigma,
                             min_abs_z, min_signal_range)


def detect_signal_extremum(ts: np.ndarray, w_lo: int, w_hi: int,
                           threshold_ratio: float = 2.0,
                           smooth_sigma: float = 5.0,
                           kind: str = 'peak',
                           min_abs_z: float = 0.5,
                           min_signal_range: float = 0.0):
    """
    Detect a prominent peak or trough in any signal within [w_lo, w_hi].

    Uses the same dual-criterion strategy as detect_j_peak (Criterion 1:
    prominence; Criterion 2: level shift).  For trough detection the signal
    is negated internally so the same peak-finding logic applies.

    Parameters
    ----------
    ts              : signal time series (numpy array)
    w_lo, w_hi      : search window boundaries
    threshold_ratio : detection threshold for both criteria (default 2.0)
    smooth_sigma    : Gaussian smoothing width in bins (default 5.0)
    kind            : ``'peak'`` (maximum) or ``'trough'`` (minimum)
    min_abs_z       : absolute amplitude gate in units of full-signal std (default 0.5)

    Returns
    -------
    has_event   : bool
    event_idx   : int   — index of the detected extremum in the original ts
    event_value : float — raw signal value at event_idx
    ratio       : float — max(prominence_ratio, level_z) test statistic
    """
    ts      = np.asarray(ts, dtype=float)
    ts_work = -ts if kind == 'trough' else ts
    return _detect_peak_core(ts_work, ts, w_lo, w_hi, threshold_ratio, smooth_sigma,
                             min_abs_z, min_signal_range)
