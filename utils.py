"""
Optical-Electrophysiology Preprocessing Utilities
==================================================

Pipeline for aligning calcium imaging (dF/F) traces with ground-truth
electrophysiological spike times for downstream spike-inference
training/validation.

Pipeline order:
    1. standardize_trace          -- resample dF/F to a uniform grid
    2. bin_discrete_spikes        -- project spikes onto frame bins
    3. fit_dataset_level_lag +
       bounded_cross_correlation_alignment
                                   -- correct hardware timing lag (fit ONE
                                      lag per source dataset, then apply it
                                      to each of that dataset's recordings)
    4. correct_dff_baseline_drift -- remove slow drift from ALREADY-COMPUTED
                                      dF/F (use compute_dynamic_dff instead
                                      only if you truly have raw fluorescence)
    5. robust_quality_control     -- SNR / spike-count based exclusion
    6. smooth_spike_train         -- Gaussian-smoothed target, in true Hz
    7. partition_recordings       -- stratified recording-level train/val split
    8. generate_sliding_windows   -- windows, built only from QC-passed data

CHANGELOG (patched review round):
    - robust_quality_control: fixed `return is_valid,` trailing-comma bug
      that returned a 1-tuple (always truthy) and silently dropped `metrics`.
    - generate_sliding_windows: now reads configurable keys (default
      'dff_clean' / 'smoothed_spike_rates') and raises a clear error instead
      of silently pulling pre-QC, pre-baseline-correction data.
    - compute_dynamic_dff: now guards against being fed an already-computed
      dF/F trace (raw fluorescence can't be meaningfully negative); added
      correct_dff_baseline_drift() as the correct tool for that case.
    - bounded_cross_correlation_alignment: unchanged behavior, but its
      docstring now says plainly that a per-recording fit conflates hardware
      lag with indicator kinetics; fit_dataset_level_lag() is the
      recommended replacement (fits one shared lag per source dataset).
    - smooth_spike_train: now multiplies by target_fs so the output is
      actually in Hz, matching its name/docstring/plot labels.
    - partition_recordings: now stratifies by a per-recording key (default
      'dataset') and uses a local np.random.Generator instead of the global
      np.random seed.

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
from scipy.ndimage import gaussian_filter1d

from typing import Dict, Tuple, List, Optional


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


def _shift_and_truncate(a, b, lag):
    """
    Shift two equal-length 1D arrays by an integer `lag` and truncate them
    to their overlapping region. Shared by find_best_lag,
    fit_dataset_level_lag, and bounded_cross_correlation_alignment so the
    shift convention only needs to be correct in one place.
    """
    if lag < 0:
        return a[-lag:], b[:lag]
    elif lag > 0:
        return a[:-lag], b[lag:]
    return a, b


def find_best_lag(dff_trace, binned_spikes, frame_rate, max_lag_sec=0.5):
    """
    Search a bounded range of integer-frame shifts and return the one that
    maximizes the Pearson correlation between `dff_trace` and
    `binned_spikes`. Pure search -- does not truncate/apply anything.

    CAVEAT (read before using this on a single recording): this maximizes
    correlation against the *entire* kinetics-smeared calcium response, not
    just a hardware sync offset -- see the docstring of
    bounded_cross_correlation_alignment for a concrete demonstration that
    this conflates indicator rise/decay kinetics with true hardware lag.
    For anything but a quick diagnostic, prefer fit_dataset_level_lag.

    Returns
    -------
    best_lag : int
        Frame shift maximizing correlation (positive = dF/F leads the
        spikes; see bounded_cross_correlation_alignment for the full sign
        convention). Falls back to 0 if every candidate lag was too flat
        to correlate.
    """
    dff_trace = np.asarray(dff_trace)
    binned_spikes = np.asarray(binned_spikes)
    max_lag_frames = int(np.ceil(max_lag_sec * frame_rate))
    best_lag, max_corr = 0, -np.inf

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        x, y = _shift_and_truncate(dff_trace, binned_spikes, lag)
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            continue
        r, _ = pearsonr(x, y)
        if r > max_corr:
            max_corr = r
            best_lag = lag

    return best_lag


# Not used in assumed hardware lag cleaned data
# also will destroy the current research question
def fit_dataset_level_lag(dff_traces, binned_spikes_list, frame_rate, max_lag_sec=0.5):
    """
    RECOMMENDED replacement for per-recording lag fitting. Fits ONE shared
    integer lag across all recordings from the same source dataset/rig.

    PROBLEM
        A genuine hardware/trigger synchronization offset is a property of
        the acquisition setup -- shared by every recording from that rig/
        session -- not of any individual neuron's response kinetics.
        Fitting a separate "best" lag per recording (as
        bounded_cross_correlation_alignment does) instead picks whatever
        shift maximizes correlation against that neuron's *entire*
        kinetics-smeared calcium response. Empirically (see module tests),
        a synthetic GCaMP6f-like kernel with ZERO injected hardware lag
        still gets "corrected" by about -67 ms, and a GCaMP6s-like kernel
        by about -133 ms -- purely from kinetics, and differently per
        indicator. That lines up with what this pipeline found on the real
        training data (recording 0: -133 ms; dataset average: -205 ms; max
        hit the +/-500 ms search bound), which is far larger and more
        variable than genuine acquisition sync offsets reported for
        similar ground-truth datasets in the literature (on the order of a
        few samples, fit once per dataset -- not hundreds of ms per
        recording). For a project specifically asking whether GCaMP6f and
        GCaMP6s generalize asymmetrically, silently applying a different,
        spike-dependent time shift to each indicator before training is a
        serious confound.

    SOLUTION
        Pool the Pearson correlation across every recording in the group at
        each candidate lag (mean correlation, ignoring recordings too flat
        to correlate at that lag) and pick the lag maximizing the group
        average. A single recording's idiosyncratic kinetics can no longer
        dominate the result; only a shift that's good across the whole
        dataset wins, consistent with what a real fixed hardware offset
        should look like.

    WHY NOT
        Still fitting per-recording: as above, conflates kinetics with lag.
        Not fitting a lag at all: if there genuinely is a small dataset-
        level sync offset, this is the more defensible way to find it.

    IMPORTANT
        This still requires ground-truth spike times, so it can only be
        computed on data where you have them (i.e. your training data).
        Do NOT compute a spike-dependent lag on the held-out test set and
        apply it there -- that leaks label information into the "held-out"
        input features. If you believe the test dataset also needs sync
        correction, that value must come from something other than its own
        spike times (e.g. a documented rig constant, or simply 0).

    Parameters
    ----------
    dff_traces : List[np.ndarray]
        One dF/F trace per recording in the group (same source dataset).
    binned_spikes_list : List[np.ndarray]
        Matching binned spike-count arrays, same lengths as `dff_traces`.
    frame_rate : float
        Sampling rate in Hz, shared by all recordings in the group.
    max_lag_sec : float, default 0.5
        Maximum lag to search, in seconds.

    Returns
    -------
    best_lag : int
        The single shared lag (in frames) to apply to every recording in
        this dataset via `_shift_and_truncate` / `bounded_cross_correlation_alignment`.
    """
    if len(dff_traces) != len(binned_spikes_list):
        raise ValueError("dff_traces and binned_spikes_list must have the same length.")

    max_lag_frames = int(np.ceil(max_lag_sec * frame_rate))
    best_lag, best_mean_corr = 0, -np.inf

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        corrs = []
        for dff_trace, binned_spikes in zip(dff_traces, binned_spikes_list):
            x, y = _shift_and_truncate(np.asarray(dff_trace), np.asarray(binned_spikes), lag)
            if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                continue
            r, _ = pearsonr(x, y)
            corrs.append(r)
        if corrs and np.mean(corrs) > best_mean_corr:
            best_mean_corr = np.mean(corrs)
            best_lag = lag

    return best_lag


# Not used in assumed hardware lag aligned data
# also will destroy the current 
def apply_lag(dff_trace, binned_spikes, lag):
    """
    Apply an already-known integer frame `lag` (e.g. from
    fit_dataset_level_lag) to a single recording's (dff_trace, binned_spikes)
    pair, without re-fitting a lag for that recording individually. Public
    wrapper around the shared shift/truncate step so dataset-level fitting
    can be used from a notebook without reaching into a private helper.

    Returns
    -------
    aligned_dff, aligned_spikes : np.ndarray
        Both truncated to their overlap after applying `lag`.
    """
    dff_trace = np.asarray(dff_trace)
    binned_spikes = np.asarray(binned_spikes)
    return _shift_and_truncate(dff_trace, binned_spikes, lag)


def bounded_cross_correlation_alignment(dff_trace, binned_spikes, frame_rate, max_lag_sec=0.5):
    """
    Detect and correct a temporal lag between one recording's optical trace
    and its binned ground-truth spike train, by shifting to maximize their
    Pearson correlation.

    HONEST LIMITATION (read this before using it as-is): this function
    fits and applies a lag using ONLY this one recording, which conflates
    genuine hardware/trigger synchronization offsets with calcium-indicator
    rise/decay kinetics -- it cannot tell them apart, because it only ever
    sees "whatever shift maximizes correlation with this neuron's response."
    A synthetic test with a realistic GCaMP kernel and ZERO injected
    hardware lag still gets "corrected" by tens to over a hundred ms,
    varying with indicator speed. For a pipeline that's meant to isolate a
    small, fixed hardware property (as the PROBLEM/WHY sections below
    describe), prefer `fit_dataset_level_lag`, which pools correlation
    across every recording in a source dataset and fits one shared lag --
    consistent with how this kind of synchronization correction is done in
    the literature for comparable ground-truth datasets (a handful of
    samples, fit once per dataset, not hundreds of ms per recording). This
    function is kept for single-recording use (e.g. quick diagnostics) and
    is exactly what `fit_dataset_level_lag` calls under the hood, via the
    shared `_shift_and_truncate` / `find_best_lag` helpers, once a lag has
    been chosen.

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
        is really just overfitting to noise. (In practice, per the
        limitation above, this bound ends up wide enough to also absorb
        indicator kinetics -- another reason to prefer the dataset-level
        fit for anything beyond a quick look.)

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

    best_lag = find_best_lag(dff_trace, binned_spikes, frame_rate, max_lag_sec)
    aligned_dff, aligned_spikes = _shift_and_truncate(dff_trace, binned_spikes, best_lag)

    return aligned_dff, aligned_spikes, best_lag



def compute_dynamic_dff(f_soma, f_neuropil=None, alpha=0.7, fps=30.0, window_sec=15.0, percentile=8):
    """
    Performs neuropil subtraction and computes a dynamic dF/F trace using a sliding percentile filter.
    Strictly avoids clipping negative values to preserve shot-noise distributions.

    ONLY VALID FOR RAW FLUORESCENCE. This dataset's `.npz` never actually
    provides raw F / neuropil traces -- only pre-computed dF/F -- so this
    function has no legitimate input to operate on for this project. It was
    previously being called with an already-computed dF/F trace as `f_soma`,
    which silently recomputed a bogus "dF/F of dF/F": the rolling percentile
    of an already ~zero-centred signal is itself close to zero, so dividing
    by it amplifies the trace by a huge, non-stationary, essentially random
    factor (tested empirically: an input with std=0.135 came out with
    std=3.34 -- a 25x blow-up -- and a range up to ~42). Raw fluorescence is
    a photon-count-like quantity and can't be meaningfully negative, so a
    guard below now refuses inputs that don't look like raw F, instead of
    silently corrupting them. If your input is already dF/F, use
    `correct_dff_baseline_drift()` instead (subtracts, not divides).

    Parameters:
    -----------
    f_soma : 1D numpy array
        Raw fluorescence trace from the somatic ROI (must be ~non-negative).
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
    f_soma = np.asarray(f_soma, dtype=float)

    # Guard: raw fluorescence (photon/ADU counts) is essentially always
    # non-negative. A meaningful fraction of negative samples is a strong,
    # cheap signal that this is already a dF/F ratio, not raw F -- exactly
    # the misuse that produced the 25x blow-up described above.
    frac_negative = np.mean(f_soma < 0)
    if frac_negative > 0.01:
        raise ValueError(
            f"{frac_negative:.1%} of f_soma is negative, which is inconsistent "
            "with raw fluorescence (photon/ADU counts can't be negative). "
            "This usually means you're passing an already-computed dF/F trace "
            "into a function meant for RAW fluorescence. If your input is "
            "already dF/F, use correct_dff_baseline_drift() instead."
        )

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


import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d

def correct_dff_baseline_drift(dff, fps=30.0, window_sec=15.0, percentile=8):
    """
    Removes slow baseline drift (e.g. residual photobleaching) from an
    ALREADY-COMPUTED dF/F trace, for use in place of compute_dynamic_dff
    when you don't have raw fluorescence (which is the case for this
    project's dataset -- only `dff` is provided, never raw F or neuropil).

    PROBLEM
        compute_dynamic_dff's baseline-subtraction-AND-DIVISION formula is
        only valid when the "baseline" is a large, slowly-varying, strictly
        positive quantity, as raw fluorescence is. Applying that same
        divide-by-baseline formula to an already-computed dF/F trace divides
        by a baseline that's already close to zero, causing unstable,
        wildly inflated output (see compute_dynamic_dff's docstring).

    SOLUTION
        Estimate the slow drift as a rolling low percentile of the dF/F
        trace itself. To avoid non-linear edge artifacts (jagged steps) 
        underneath dense spike bursts, smooth this percentile estimate with 
        a uniform filter. Finally, SUBTRACT it (an additive correction).

    WHY NOT
        Re-deriving "dF/F of dF/F" via compute_dynamic_dff: mathematically
        invalid for the reason above.

    Parameters
    ----------
    dff : np.ndarray
        1D array, an already-computed dF/F trace (may contain negative
        values -- that's expected and fine here, unlike for raw F).
    fps : float
        Sampling rate in Hz.
    window_sec : float
        Sliding window size in seconds for drift estimation.
    percentile : int
        Percentile used to estimate the slow-drift floor (default 8th).

    Returns
    -------
    dff_corrected : np.ndarray
        The drift-corrected dF/F trace (dff - drift). Negative values are
        preserved, not clipped, consistent with the rest of this pipeline.
    drift : np.ndarray
        The estimated (and smoothed) slow-drift component that was subtracted.
    """
    dff = np.asarray(dff, dtype=float)

    # 1. Calculate base window frames
    window_frames = int(window_sec * fps)
    if window_frames % 2 == 0:
        window_frames += 1

    # 2. Extract the raw non-linear baseline via percentile filter
    raw_drift = percentile_filter(dff, percentile=percentile, size=window_frames, mode='reflect')
    
    # 3. Smooth the jagged baseline to prevent edge-artifacts under spikes
    # A smoothing window of 1/4th the main window beautifully rounds the steps
    smooth_frames = max(1, window_frames // 4)
    smoothed_drift = uniform_filter1d(raw_drift, size=smooth_frames, mode='reflect')

    # 4. Apply additive correction
    dff_corrected = dff - smoothed_drift

    return dff_corrected, smoothed_drift


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

    # 99th Percentile Peak
    baseline_median = np.median(dff)
    robust_peak = np.percentile(dff, 99)
    
    signal_est = robust_peak - baseline_median
    
    # Epsilon prevents ZeroDivisionError if a trace is perfectly flat (artifact)
    snr = signal_est / (noise_est + np.finfo(float).eps)
    
    is_valid = bool((total_spikes >= min_spikes) and (snr >= snr_threshold))
    
    metrics = {
        "total_spikes": total_spikes,
        "snr": snr,
        "noise_level": noise_est
    }

    # BUG FIX: this used to be `return is_valid, ` -- a 1-tuple due to the
    # trailing comma. `metrics` was silently discarded, and because a
    # non-empty tuple is always truthy in Python, `if is_valid:` at the call
    # site was always True regardless of the actual QC outcome (this is why
    # the notebook reported 0 exclusions out of 83 recordings). Callers must
    # now unpack both values: `is_valid, metrics = robust_quality_control(...)`.
    return is_valid, metrics



def smooth_spike_train(spike_counts: np.ndarray, target_fs: float = 30.0, sigma_sec: float = 0.05) -> np.ndarray:
    """
    Convolves a discrete spike count array with a Gaussian kernel to create
    a continuous spike RATE (in Hz) target for 1D-CNN training.

    BUG FIX: `gaussian_filter1d` uses a normalized (mass-preserving) kernel,
    so convolving it with per-frame spike COUNTS gives you smoothed counts
    per FRAME, not a rate per SECOND -- summing the output recovers the
    original total spike count, not that count times the frame rate. The
    previous version returned this un-rescaled quantity while calling it
    "smoothed_rates" / "Spike Rate (Hz)". That's harmless for training as
    long as target_fs is fixed everywhere (a constant scale factor), but it
    is not actually in Hz, which matters if you ever compare against
    Cascade's own (Hz-scaled) outputs or mix frame rates. Multiplying by
    target_fs below converts counts/frame -> counts/second (Hz).

    Parameters:
    -----------
    spike_counts : np.ndarray
        1D array of discrete binned spike counts (e.g., 0, 1, 2) at the target frame rate.
    target_fs : float
        The sampling frequency of the array in Hz. Default is 30.0 Hz.
    sigma_sec : float
        The standard deviation of the Gaussian kernel in seconds. 
        Default is 0.05s, calibrated for 30Hz recordings (matches Cascade's
        own published default at this frame rate).

    Returns:
    --------
    smoothed_rates_hz : np.ndarray
        1D array of continuous smoothed spike RATES in Hz, preserving the
        original shape.
    """
    # 1. Convert physical time (seconds) to discrete frames based on sampling rate
    sigma_frames = sigma_sec * target_fs
    
    # 2. Apply zero-phase 1D Gaussian convolution
    # mode='constant' with cval=0.0 ensures we assume zero biological activity 
    # outside the recording boundaries, avoiding artificial edge padding artifacts.
    smoothed_counts_per_frame = gaussian_filter1d(
        input=spike_counts.astype(float),
        sigma=sigma_frames,
        mode='constant',
        cval=0.0 
    )

    # 3. Convert counts/frame -> counts/second (Hz). See BUG FIX note above.
    smoothed_rates_hz = smoothed_counts_per_frame * target_fs

    return smoothed_rates_hz



def partition_recordings(
    dataset: Dict[str, dict],
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_key: Optional[str] = "dataset",
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Partitions calcium imaging data strictly at the recording/dataset level.
    Must be executed BEFORE any sliding window operations to prevent temporal leakage.

    CHANGES FROM THE ORIGINAL VERSION
        - Stratifies by `stratify_key` (default: each recording's source
          'dataset' label) so DS09/DS10/DS14/DS15 -- and therefore GCaMP6f
          vs GCaMP6s -- are each represented proportionally in both splits.
          A plain pooled-random split over only ~83 recordings (~12 going
          to validation) could easily land a validation set skewed toward
          one indicator by chance, which is a real problem when your
          research question is specifically about 6f-vs-6s asymmetry: you
          would not be able to tell a genuine generalization gap from
          validation-set sampling noise. Falls back to an unstratified
          split if `stratify_key` is missing from the data, or is None.
        - Uses a local np.random.Generator instead of the global
          np.random.seed(), so calling this doesn't silently change the
          random state for anything else in the notebook that also uses
          np.random.

    Parameters:
    -----------
    dataset : dict
        The master dictionary containing processed recordings (e.g., phase2_aligned_data).
    val_ratio : float
        Proportion of recordings to allocate to the validation set (applied
        within each stratum).
    seed : int
        Random state for reproducibility.
    stratify_key : str or None
        Per-recording dict key to stratify on (default 'dataset'). Pass
        None to fall back to a plain pooled random split.

    Returns:
    --------
    train_set, val_set : Tuple of disjoint dictionaries.
    """
    rng = np.random.default_rng(seed)
    recording_ids = list(dataset.keys())

    can_stratify = stratify_key is not None and all(
        stratify_key in dataset[rid] for rid in recording_ids
    )

    if can_stratify:
        groups: Dict[str, list] = {}
        for rid in recording_ids:
            groups.setdefault(dataset[rid][stratify_key], []).append(rid)
    else:
        groups = {"_all_": recording_ids}

    train_ids, val_ids = [], []
    for _, ids in groups.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    # Reconstruct isolated datasets
    train_set = {rid: dataset[rid] for rid in train_ids}
    val_set = {rid: dataset[rid] for rid in val_ids}

    return train_set, val_set

def generate_sliding_windows(
    isolated_dataset: Dict[str, dict],
    window_size: int = 64,
    dff_key: str = "dff_clean",
    target_key: str = "smoothed_spike_rates",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps continuous recordings to a Hankel matrix via a sliding window.
    Applies ONLY to a pre-isolated subset to guarantee out-of-distribution evaluation.

    BUG FIX: this used to hard-code `data['aligned_dff']` -- a Phase-2 key
    -- so calling it on `partition_recordings(phase2_aligned_data, ...)`
    silently trained on the pre-QC, pre-baseline-correction trace and never
    touched Phase 3's output (`phase3_clean_data['dff_clean']`) at all,
    regardless of whether QC passed or failed. `dff_key`/`target_key` are
    now parameters (defaulting to the Phase-3-correct names) and a missing
    key now raises a clear error instead of that silent misuse. Pass
    `isolated_dataset` built from `partition_recordings(phase3_clean_data, ...)`
    once Phase 4's smoothed targets have been merged onto `phase3_clean_data`
    (not left on `phase2_aligned_data`, or this will still miss them).

    Parameters:
    -----------
    isolated_dataset : dict
        A disjoint subset (e.g., train_set) returned by partition_recordings,
        built from your QC-filtered, baseline-corrected data (phase3_clean_data)
        with Phase 4's smoothed spike-rate targets merged in.
    window_size : int
        The temporal receptive field of the 1D-CNN (default: 64 frames).
    dff_key : str
        Per-recording dict key holding the input dF/F trace to window.
    target_key : str
        Per-recording dict key holding the smoothed spike-rate target.

    Returns:
    --------
    X : np.ndarray
        Shape (N_samples, window_size). The input delta F/F windows.
    Y : np.ndarray
        Shape (N_samples,). The (approximately) center-aligned ground truth
        smoothed spike rate -- for an even window_size the target frame is
        one index right of the true midpoint, an unavoidable +/-0.5 frame
        rounding with even-length windows.
    """
    X, Y = [], []
    half_window = window_size // 2

    for rec_id, data in isolated_dataset.items():
        if dff_key not in data or target_key not in data:
            raise KeyError(
                f"Recording {rec_id!r} is missing '{dff_key}' or '{target_key}'. "
                "generate_sliding_windows expects QC-filtered, baseline-corrected "
                "data (phase3_clean_data) with Phase 4's smoothed targets merged "
                "onto it -- make sure you partitioned phase3_clean_data (not "
                "phase2_aligned_data), and that the smoothing step wrote onto "
                "the same dict you're passing in here."
            )
        dff = data[dff_key]
        spikes = data[target_key]

        # Skip recordings smaller than the receptive field
        if len(dff) <= window_size:
            continue

        # Stride = 1 step
        for t in range(half_window, len(dff) - half_window):
            X.append(dff[t - half_window : t + half_window])
            Y.append(spikes[t])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32


def scale_features(X_train, X_val, X_test):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Crucially, it computes the metrics ONLY on the training set to prevent 
    data leakage into the test set.
    """
    print("--- Feature Scaling (Z-Scoring) ---")
    
    # 1. Calculate GLOBAL mean and std from the TRAINING set only.
    # We take the global scalar mean, NOT axis=0. If you use axis=0, you  
    # normalize each time-bin independently, which completely destroys the 
    # temporal shape of your calcium transients!
    mu_train = np.mean(X_train)
    sigma_train = np.std(X_train)
    
    # Guard against division by zero (rare, but good defensive programming)
    if sigma_train == 0:
        sigma_train = 1e-8
        
    print(f"Training Mean: {mu_train:.4f} | Training Std: {sigma_train:.4f}")

    # 2. Apply the exact same transformation to all datasets
    X_train_scaled = (X_train - mu_train) / sigma_train
    X_val_scaled = (X_val - mu_train) / sigma_train
    X_test_scaled = (X_test - mu_train) / sigma_train

    return X_train_scaled, X_val_scaled, X_test_scaled




    ####### EVALUATION _ TODO

    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Formally evaluates 1D-CNN predictions against ground-truth spike rates.
    
    PROBLEM
        Loss functions like MSE are necessary for backpropagation but are 
        biologically uninterpretable. An MSE of 0.04 doesn't tell us if the 
        model actually detected the spikes.
        
    SOLUTION
        Calculate Pearson Correlation (r) to measure kinetic shape alignment, 
        and F1-Score to measure exact discrete event detection.
        
    Parameters
    ----------
    y_true : np.ndarray
        1D array of ground-truth smoothed spike rates.
    y_pred : np.ndarray
        1D array of network predictions.
    threshold : float
        The cutoff value to binarize continuous predictions into discrete spikes.
        
    Returns
    -------
    metrics : dict
        Contains 'pearson_r', 'f1_score', and 'mse'.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import f1_score
    
    # 1. Pearson Correlation (Kinetic Shape & Timing)
    # Returns (statistic, p-value), we only need the statistic
    r_val, _ = pearsonr(y_true, y_pred)
    
    # 2. Mean Squared Error (Magnitude accuracy)
    mse = np.mean((y_true - y_pred)**2)
    
    # 3. F1-Score (Discrete Detection)
    # Binarize the ground truth: if the smoothed rate is > 0, a spike occurred nearby
    y_true_binary = (y_true > 0).astype(int)
    # Binarize the prediction: if the network outputs a rate > threshold, call it a spike
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # F1 is the harmonic mean of precision and recall. 
    # zero_division=0 prevents warnings if the network predicts absolutely nothing.
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        "pearson_r": r_val,
     