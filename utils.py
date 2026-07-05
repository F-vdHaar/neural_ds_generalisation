"""
Optical-Electrophysiology Preprocessing Utilities
==================================================

Pipeline for aligning calcium imaging (dF/F) traces with ground-truth
electrophysiological spike times for downstream spike-inference
training/validation.

Pipeline order:
    1. standardize_trace                  -- resample dF/F to a uniform grid
    2. bin_discrete_spikes                -- project spikes onto frame bins
    3. bounded_cross_correlation_alignment -- correct hardware timing lag

Each function's docstring follows a uniform template:
    PROBLEM  -- what goes wrong without this step
    SOLUTION -- what the function does about it
    WHY      -- the reasoning behind that approach
    HOW      -- the concrete algorithm
    WHY NOT  -- alternatives considered and why they were rejected
"""

import numpy as np
import warnings
from scipy.signal import cheby1, sosfiltfilt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from scipy.ndimage import percentile_filter



def standardize_trace(dff, t, target_fs=30.0, aa_safety_factor=0.9):
    """
    Resample a continuous dF/F trace onto a uniform time grid at `target_fs`.

    PROBLEM
        Raw dF/F traces are rarely sampled at a perfectly uniform rate
        (frame drops, hardware jitter). Downstream steps that assume a
        fixed dt (spike binning, cross-correlation, model input) will
        silently misbehave if fed a trace that is only approximately
        evenly sampled.

    SOLUTION
        Low-pass (anti-alias) filter the signal, then linearly interpolate
        it onto an exact, evenly-spaced time vector at `target_fs`.

    WHY
        Interpolating alone can alias high-frequency noise or motion
        artifacts down into the signal band. Filtering first removes
        content above the new Nyquist limit so the resampled trace is a
        faithful, band-limited version of the original -- but only when
        actually downsampling.

    HOW
        1. Estimate the original sampling rate from the median dt of `t`.
        2. If downsampling (fs_original > target_fs), design an 8th-order
           Chebyshev Type I low-pass filter (SOS form, for numerical
           stability) with cutoff at `aa_safety_factor * target_fs / 2`.
        3. Apply it with zero phase distortion (sosfiltfilt).
        4. Build the exact target time vector and linearly interpolate
           the (possibly filtered) signal onto it.

    WHY NOT
        - Cutoff at exactly target_fs/2, no safety factor: Chebyshev
          filters roll off gradually, not instantly, so frequencies just
          below the exact Nyquist edge are only partially attenuated. A
          small margin (default 0.9x) trades a little bandwidth for real
          protection against aliasing.
        - Always filtering: if fs_original <= target_fs (upsampling),
          there is no aliasing risk, and scipy will raise an error if
          asked for a cutoff at or above the input Nyquist frequency.
          The filter step is skipped in that case.
        - Cubic/spline interpolation: more accurate for smooth traces,
          but can overshoot near sharp transients (e.g., true calcium
          transients) with noisy data. Linear is the safer default;
          change `kind=` below if your data warrants it.

    Parameters
    ----------
    dff : np.ndarray
        1D raw delta F/F trace.
    t : np.ndarray
        1D timestamps for `dff`, in seconds (need not be uniform).
    target_fs : float, default 30.0
        Desired output sampling rate, in Hz.
    aa_safety_factor : float, default 0.9
        Fraction of the target Nyquist frequency actually used as the
        filter cutoff, to leave margin against imperfect roll-off.

    Returns
    -------
    dff_standard : np.ndarray
        The resampled trace at exactly `target_fs`.
    t_standard : np.ndarray
        The corresponding uniform timestamps.
    """
    # 1. Determine original sampling rate
    dt_original = np.median(np.diff(t))
    fs_original = 1.0 / dt_original

    # 2. Anti-alias filter -- only meaningful when downsampling
    nyq_target = target_fs / 2.0
    if fs_original > target_fs:
        cutoff = aa_safety_factor * nyq_target
        sos = cheby1(N=8, rp=0.05, Wn=cutoff, btype='low', fs=fs_original, output='sos')
        dff_filtered = sosfiltfilt(sos, dff)
    else:
        # Upsampling (or equal rate): no aliasing risk. Filtering here
        # would either be a no-op or raise (Wn must be < fs_original / 2).
        dff_filtered = dff

    # 3. Exact, standardized time vector
    t_standard = np.arange(t[0], t[-1], 1.0 / target_fs)

    # 4. Interpolate onto the new grid
    interpolator = interp1d(t, dff_filtered, kind='linear', bounds_error=False, fill_value="extrapolate")
    dff_standard = interpolator(t_standard)

    # FIX: original code returned `dff_standard,` (a 1-tuple due to the
    # trailing comma), silently dropping t_standard from the return value.
    return dff_standard, t_standard


def bin_discrete_spikes(spike_times, frame_times, warn_on_drop=True):
    """
    Project continuous electrophysiological spike times into discrete
    per-frame spike counts, aligned to optical frame timestamps.

    PROBLEM
        Ground-truth spikes are timestamped continuously (ephys clock),
        but the optical signal is sampled discretely (camera frames). To
        compare or train against the two, every spike must be assigned to
        exactly one frame, with none silently lost or double-counted.

    SOLUTION
        Build bin edges at the midpoints between consecutive frame
        timestamps (plus extrapolated half-width edges at the two ends),
        then histogram the spike times into those edges.

    WHY
        Midpoint edges (rather than, e.g., [frame, frame + dt)) guarantee
        every point in time belongs to the bin of its temporally nearest
        frame -- the least-biased assignment when frame intervals aren't
        perfectly uniform.

    HOW
        1. Validate that `frame_times` is strictly increasing (required
           for well-defined, non-overlapping bins).
        2. Compute interior edges as true local midpoints
           (frame[i] + diff[i]/2); extrapolate the first/last edges using
           the median dt.
        3. Histogram `spike_times` into those edges.
        4. Explicitly verify spike conservation and raise (not assert) if
           the invariant is violated.
        5. Warn if any spikes fell outside the recording window, so the
           exclusion is visible rather than silent.

    WHY NOT
        - `assert` for the conservation check (as in the original code):
          assertions are stripped entirely when Python runs with
          `-O` / `PYTHONOPTIMIZE=1`. For a data-integrity check in a
          shared analysis pipeline, that means the safety net can vanish
          depending on how the script happens to be invoked. An explicit
          `raise ValueError` cannot be disabled this way.
        - Silently dropping out-of-window spikes: numerically harmless,
          but it hides information you'll want for a Methods section
          (e.g., "12 of 4,502 spikes fell outside the imaging window").
        - Tolerating duplicate/non-increasing frame timestamps: this
          usually signals an upstream frame-time bug (e.g., an
          incomplete dropped-frame correction). Failing fast with a
          clear message beats silently producing degenerate bins.

    Parameters
    ----------
    spike_times : np.ndarray
        1D array of ground-truth action potential times (seconds).
    frame_times : np.ndarray
        1D array of optical frame timestamps (seconds), strictly
        increasing.
    warn_on_drop : bool, default True
        If True, emit a warning reporting how many spikes fell outside
        [edges[0], edges[-1]] and were excluded.

    Returns
    -------
    binned_spikes : np.ndarray
        1D array of exact spike counts per frame, same length as
        `frame_times`.
    """
    frame_times = np.asarray(frame_times)
    spike_times = np.asarray(spike_times)

    # 0. Precondition: frames must be strictly increasing for edges to be
    #    well-defined and non-overlapping.
    frame_diffs = np.diff(frame_times)
    if not np.all(frame_diffs > 0):
        raise ValueError(
            "frame_times must be strictly increasing. Found "
            f"{int(np.sum(frame_diffs <= 0))} non-positive interval(s) -- "
            "check for duplicate or out-of-order frame timestamps upstream."
        )

    # 1. Bin edges at true local midpoints, extrapolated at the ends
    dt = np.median(frame_diffs)
    edges = np.concatenate([
        [frame_times[0] - dt / 2],
        frame_times[:-1] + frame_diffs / 2,
        [frame_times[-1] + dt / 2],
    ])

    binned_spikes, _ = np.histogram(spike_times, bins=edges)

    # 2. Explicit conservation check (raise, not assert -- see WHY NOT)
    valid_spike_mask = (spike_times >= edges[0]) & (spike_times <= edges[-1])
    expected_spike_total = np.sum(valid_spike_mask)
    actual_binned_total = np.sum(binned_spikes)

    if expected_spike_total != actual_binned_total:
        raise ValueError(
            "Event loss detected during binning: expected "
            f"{expected_spike_total} spikes within the temporal window, "
            f"but the binned array contains {actual_binned_total}."
        )

    # 3. Surface (rather than silently drop) out-of-window spikes
    n_dropped = spike_times.size - expected_spike_total
    if warn_on_drop and n_dropped > 0:
        pct = 100 * n_dropped / spike_times.size
        warnings.warn(
            f"{n_dropped} of {spike_times.size} spikes ({pct:.1f}%) fell "
            "outside the frame recording window and were excluded from "
            "binning."
        )

    return binned_spikes


def bounded_cross_correlation_alignment(dff_trace, binned_spikes, frame_rate, max_lag_sec=0.5):
    """
    Detect and correct hardware-induced temporal lag between the optical
    trace and the binned ground-truth spike train.

    PROBLEM
        Optical and electrical acquisition systems are triggered
        independently and can have a small, fixed timing offset (cable
        delay, trigger jitter, clock drift). Left uncorrected, this lag
        biases any frame-by-frame comparison between the two modalities.

    SOLUTION
        Search a bounded range of integer-frame shifts and pick the shift
        that maximizes the Pearson correlation between the (shifted)
        optical trace and the binned spike counts.

    WHY
        The true lag is a fixed hardware property, not a free parameter
        to fit per analysis, so it should be small and stable. Bounding
        the search (`max_lag_sec`, default 500 ms) prevents the optimizer
        from "correcting" onto a physiologically implausible shift that
        is really just overfitting to noise.

    HOW
        1. Convert `max_lag_sec` to an integer number of frames.
        2. For each candidate lag in [-max_lag_frames, +max_lag_frames],
           shift and truncate the two series to overlap, then compute
           their Pearson correlation (skipping shifts that leave a
           constant/flat segment, which cannot support a correlation).
        3. Keep the lag with the highest correlation.
        4. Apply that shift and return the truncated, aligned pair.

    WHY NOT
        - FFT-based cross-correlation (e.g. `scipy.signal.correlate`):
          faster, but it computes covariance rather than Pearson's
          normalized correlation, so it doesn't automatically account
          for DC offset or scale differences between dF/F and spike
          counts. The direct per-lag Pearson loop trades some speed for
          that normalization and for interpretability.
        - An unbounded lag search: without `max_lag_sec`, the optimizer
          could lock onto a spurious, physiologically meaningless shift,
          especially in short or noisy recordings.

    Parameters
    ----------
    dff_trace : np.ndarray
        1D downsampled delta F/F optical trace.
    binned_spikes : np.ndarray
        1D discrete binned action-potential counts, same length as
        `dff_trace`.
    frame_rate : float
        Current sampling rate, in Hz (e.g., 30.0 or 7.5).
    max_lag_sec : float, default 0.5
        Maximum physiologically plausible lag to search, in seconds.

    Returns
    -------
    aligned_dff : np.ndarray
        The dF/F trace, truncated to overlap after the best-lag shift.
    aligned_spikes : np.ndarray
        The spike counts, truncated to overlap after the best-lag shift.
    best_lag : int
        The optimal shift, in frames (positive = dF/F leads the spikes).
    """
    dff_trace = np.asarray(dff_trace)
    binned_spikes = np.asarray(binned_spikes)

    if dff_trace.shape[0] != binned_spikes.shape[0]:
        raise ValueError(
            "dff_trace and binned_spikes must be the same length before "
            f"alignment (got {dff_trace.shape[0]} vs {binned_spikes.shape[0]})."
        )

    max_lag_frames = int(np.ceil(max_lag_sec * frame_rate))
    best_lag = 0
    max_corr = -np.inf

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            x = dff_trace[-lag:]
            y = binned_spikes[:lag]
        elif lag > 0:
            x = dff_trace[:-lag]
            y = binned_spikes[lag:]
        else:
            x = dff_trace
            y = binned_spikes

        # Ignore completely flat segments to avoid division by zero
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            continue

        r, _ = pearsonr(x, y)
        if r > max_corr:
            max_corr = r
            best_lag = lag

    # Apply optimal shift and truncate overlapping tails
    if best_lag < 0:
        aligned_dff = dff_trace[-best_lag:]
        aligned_spikes = binned_spikes[:best_lag]
    elif best_lag > 0:
        aligned_dff = dff_trace[:-best_lag]
        aligned_spikes = binned_spikes[best_lag:]
    else:
        aligned_dff = dff_trace
        aligned_spikes = binned_spikes

    return aligned_dff, aligned_spikes, best_lag



def compute_dynamic_dff(f_soma, f_neuropil=None, alpha=0.7, fps=30.0, window_sec=15.0, percentile=8):
    """
    Performs neuropil subtraction and computes a dynamic dF/F trace using a sliding percentile filter.
    Strictly avoids clipping negative values to preserve shot-noise distributions.
    
    Parameters:
    -----------
    f_soma : 1D numpy array
        Raw fluorescence trace from the somatic ROI.
    f_neuropil : 1D numpy array or None
        Raw fluorescence trace from the surrounding neuropil ROI.
    alpha : float
        Neuropil contamination coefficient (default 0.7).
    fps : float
        Sampling rate in Hz.
    window_sec : float
        Sliding window size in seconds for baseline estimation.
    percentile : int
        Percentile to use for the baseline calculation (default 8th percentile).
        
    Returns:
    --------
    dff : 1D numpy array
        The normalized dF/F trace.
    f0 : 1D numpy array
        The computed dynamic baseline.
    """
    # 1. Neuropil Subtraction
    if f_neuropil is not None:
        f_corrected = f_soma - (alpha * f_neuropil)
    else:
        # If dataset only provides pre-subtracted traces
        f_corrected = np.copy(f_soma)
        
    # 2. Dynamic Baseline Estimation
    window_frames = int(window_sec * fps)
    
    # Ensure window is odd for symmetric filtering padding
    if window_frames % 2 == 0:
        window_frames += 1 
        
    f0 = percentile_filter(f_corrected, percentile=percentile, size=window_frames, mode='reflect')
    
    # 3. Compute dF/F
    # We use absolute value of F0 in the denominator to prevent gradient inversions 
    # if extreme neuropil subtraction temporarily pushes the baseline below zero.
    # Epsilon prevents ZeroDivisionError on perfectly flat artifact regions.
    epsilon = np.finfo(float).eps
    dff = (f_corrected - f0) / (np.abs(f0) + epsilon)
    
    return dff, f0




def robust_quality_control(dff, spikes, fps, min_spikes=5, snr_threshold=2.5):
    """
    Evaluates if a trace meets the required quality thresholds for gradient descent.
    
    Parameters:
    -----------
    dff : 1D numpy array
        The normalized dF/F trace.
    spikes : 1D numpy array
        Discrete vector of binned ground truth spikes.
    fps : float
        The sampling rate of the recording in Hz (required for Cascade noise metric).
    min_spikes : int
        Minimum number of action potentials required in the recording.
    snr_threshold : float
        Minimum Signal-to-Noise Ratio.
        
    Returns:
    --------
    is_valid : bool
        True if the recording passes Quality Control.
    metrics : dict
        Dictionary containing calculated SNR and total spikes.
    """
    total_spikes = np.sum(spikes)

    
    # FIX CASCADE uses MASD
    # Estimate baseline noise (nu) using Median Absolute Deviation (MAD)
    # MAD is robust to the large positive skew caused by actual calcium transients
   # median_dff = np.median(dff)
    # mad = np.median(np.abs(dff - median_dff))
    
    # Convert MAD to standard deviation equivalent for normal distribution
    # noise_est = mad * 1.4826



    # Cascade's standardized noise metric (Rupprecht et al., 2021)
    # Frame-to-frame dF/F fluctuation normalized by frame rate
    successive_diffs = np.diff(dff)
    
    # Calculate MASD and normalize by sqrt(fps) to match Cascade standard
    noise_est = np.median(np.abs(successive_diffs)) / np.sqrt(fps)

    # Signal variance vs Noise variance approximation
    signal_est = np.std(dff)
    
    # Epsilon prevents ZeroDivisionError if a trace is perfectly flat (artifact)
    snr = signal_est / (noise_est + np.finfo(float).eps)
    
    is_valid = bool((total_spikes >= min_spikes) and (snr >= snr_threshold))
    
    metrics = {
        "total_spikes": total_spikes,
        "snr": snr,
        "noise_level": noise_est
    }
    
    return is_valid, metrics