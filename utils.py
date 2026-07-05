import numpy as np
from scipy.signal import cheby1, sosfiltfilt
from scipy.interpolate import interp1d

def standardize_trace(dff, t, target_fs=30.0):
    """
    Standardizes a continuous trace to an EXACT target frequency using 
    strict zero-phase, numerically stable anti-aliasing (sosfiltfilt) 
    followed by interpolation.
    """
    # 1. Determine original sampling rate
    dt_original = np.median(np.diff(t))
    fs_original = 1.0 / dt_original
    
    # 2. Design the Anti-Aliasing Filter
    # We must eliminate frequencies above the Nyquist limit of our new target rate
    # For a 30 Hz target, the Nyquist limit is 15 Hz.
    nyq_target = target_fs / 2.0
    
    # Chebyshev Type I low-pass filter (8th order, 0.05 dB ripple)
    # BEST PRACTICE: Using output='sos' (Second-Order Sections) guarantees 
    # numerical stability, avoiding the floating-point errors of 'b, a' polynomials.
    sos = cheby1(N=8, rp=0.05, Wn=nyq_target, btype='low', fs=fs_original, output='sos')
    
    # Apply STRICT ZERO-PHASE filtering using the stable sos format
    dff_filtered = sosfiltfilt(sos, dff)
        
    # 3. Create the exact, standardized 30.0 Hz time vector
    t_standard = np.arange(t[0], t[-1], 1.0 / target_fs)
    
    # 4. Interpolate the safe, anti-aliased signal onto the exact new timepoints
    interpolator = interp1d(t, dff_filtered, kind='linear', bounds_error=False, fill_value="extrapolate")
    dff_standard = interpolator(t_standard)
    
    return dff_standard, t_standard