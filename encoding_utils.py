import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq

def bandpass_filter(data, low_freq, high_freq, fs=100, order=4):
    """
    Bandpass filter for specific frequency bands
    Critical for isolating Parkinson's tremor (4-6 Hz)
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered

def compute_spectral_power(data, fs=100, bands=None):
    """
    Compute power in specific frequency bands
    Returns dict of band powers for feature engineering
    """
    if bands is None:
        bands = {
            'rest': (0, 1),      # <1 Hz (stillness)
            'postural': (1, 4),   # 1-4 Hz (postural tremor)
            'tremor': (4, 6),     # 4-6 Hz (Parkinson's resting tremor)
            'essential': (6, 10), # 6-10 Hz (essential tremor)
            'movement': (0.5, 3)  # 0.5-3 Hz (voluntary movement)
        }
    
    # Handle 1D or 2D data
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples = data.shape[0]
    freqs = fftfreq(n_samples, 1/fs)
    spectrum = np.abs(fft(data, axis=0)) ** 2
    
    # Only use positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    spectrum = spectrum[pos_mask]
    
    band_powers = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_powers[name] = np.sum(spectrum[mask], axis=0)
        else:
            band_powers[name] = np.zeros(data.shape[1])
    
    return band_powers

def frequency_aware_encoding(data, fs=100, tremor_band=(4, 6), gain=8):
    """
    IMPROVED ENCODING: Frequency-aware spike generation
    
    Steps:
    1. Bandpass filter for tremor frequency (4-6 Hz)
    2. Compute instantaneous amplitude (envelope)
    3. Convert to spikes using latency + rate coding
    
    Returns: Spike train [time_steps, features]
    """
    time_steps, features = data.shape
    spikes = torch.zeros(time_steps, features)
    
    for f in range(features):
        channel_data = data[:, f]
        
        # Step 1: Bandpass filter for tremor band
        tremor_signal = bandpass_filter(channel_data, tremor_band[0], tremor_band[1], fs)
        
        # Step 2: Compute envelope (instantaneous amplitude)
        analytic = signal.hilbert(tremor_signal)
        envelope = np.abs(analytic)
        
        # Step 3: Smooth envelope for stability
        envelope = signal.savgol_filter(envelope, window_length=11, polyorder=2)
        
        # Step 4: Normalize to [0, 1] with adaptive threshold
        envelope_norm = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)
        
        # Step 5: Latency + Rate coding
        # Higher amplitude → earlier spike in each 100ms window
        window_size = int(fs * 0.1)  # 100ms windows
        for start in range(0, time_steps, window_size):
            end = min(start + window_size, time_steps)
            window_envelope = envelope_norm[start:end]
            
            if len(window_envelope) == 0:
                continue
            
            # Find peak in window (latency coding)
            peak_idx = np.argmax(window_envelope)
            spike_probability = min(window_envelope[peak_idx] * gain, 0.95)  # Cap at 0.95
            
            # Rate coding: probability of spike at peak
            if np.random.random() < spike_probability:
                spikes[start + peak_idx, f] = 1.0
            
            # Additional spikes based on amplitude (rate coding)
            expected_spikes = int(spike_probability * 2)
            for _ in range(expected_spikes):
                rand_idx = np.random.randint(start, end)
                if np.random.random() < window_envelope[rand_idx - start] * gain * 0.5:
                    spikes[rand_idx, f] = 1.0
    
    return spikes

def compute_asymmetry_features(left_data, right_data, fs=100):
    """
    Compute bilateral asymmetry features (CRITICAL for early PD detection)
    
    Returns: Asymmetry features [time_steps, 3]
    - Channel 0: Power asymmetry (4-6 Hz band)
    - Channel 1: Amplitude asymmetry (envelope difference)
    - Channel 2: Phase asymmetry (timing difference)
    """
    time_steps = left_data.shape[0]
    asymmetry = np.zeros((time_steps, 3))
    
    # Use X-axis (most sensitive for tremor)
    left_x = left_data[:, 0]
    right_x = right_data[:, 0]
    
    # Compute spectral power for each side using sliding windows
    window_size = min(fs * 2, time_steps)  # 2-second windows
    step_size = fs // 2  # 0.5-second steps
    
    power_asym = np.zeros(time_steps)
    amp_asym = np.zeros(time_steps)
    phase_asym = np.zeros(time_steps)
    
    for center in range(0, time_steps, step_size):
        start = max(0, center - window_size // 2)
        end = min(time_steps, center + window_size // 2)
        
        if end - start < fs:  # Need at least 1 second of data
            continue
        
        left_window = left_x[start:end]
        right_window = right_x[start:end]
        
        # 1. Power asymmetry in tremor band
        left_powers = compute_spectral_power(left_window, fs)
        right_powers = compute_spectral_power(right_window, fs)
        
        left_tremor = left_powers.get('tremor', 0)
        right_tremor = right_powers.get('tremor', 0)
        
        if isinstance(left_tremor, np.ndarray):
            left_tremor = left_tremor[0]
            right_tremor = right_tremor[0]
        
        power_asym_center = np.abs(left_tremor - right_tremor) / (left_tremor + right_tremor + 1e-8)
        
        # 2. Amplitude asymmetry (envelope)
        left_analytic = signal.hilbert(left_window)
        right_analytic = signal.hilbert(right_window)
        left_envelope = np.abs(left_analytic)
        right_envelope = np.abs(right_analytic)
        
        amp_asym_center = np.abs(np.mean(left_envelope) - np.mean(right_envelope)) / (np.mean(left_envelope) + np.mean(right_envelope) + 1e-8)
        
        # 3. Phase asymmetry (cross-correlation lag)
        # Positive = left leads, Negative = right leads
        correlation = np.correlate(left_window - left_window.mean(), 
                                   right_window - right_window.mean(), 
                                   mode='full')
        lag = np.argmax(correlation) - len(left_window) + 1
        phase_asym_center = np.tanh(lag / 10)  # Normalize to [-1, 1]
        
        # Fill the window with the computed values
        power_asym[start:end] = power_asym_center
        amp_asym[start:end] = amp_asym_center
        phase_asym[start:end] = phase_asym_center
    
    # Smooth the asymmetry features
    asymmetry[:, 0] = signal.savgol_filter(power_asym, window_length=11, polyorder=2)
    asymmetry[:, 1] = signal.savgol_filter(amp_asym, window_length=11, polyorder=2)
    asymmetry[:, 2] = signal.savgol_filter(phase_asym, window_length=11, polyorder=2)
    
    return asymmetry

def improved_context_encoding(left_data, right_data, fs=100):
    """
    IMPROVED CONTEXT: Full-window PSD analysis (not just first 100 samples)
    
    Returns: Context features [time_steps, 3]
    - Channel 0: Rest probability
    - Channel 1: Postural probability
    - Channel 2: Kinetic probability
    """
    time_steps = left_data.shape[0]
    context = np.zeros((time_steps, 3))
    
    # Combine left + right for context estimation (use X and Y axes)
    combined_accel_x = np.mean([left_data[:, 0], right_data[:, 0]], axis=0)
    combined_accel_y = np.mean([left_data[:, 1], right_data[:, 1]], axis=0)
    
    # Use sliding windows for time-varying context
    window_size = min(fs * 2, time_steps)  # 2-second windows
    step_size = fs // 4  # 0.25-second steps for smooth transitions
    
    for center in range(0, time_steps, step_size):
        start = max(0, center - window_size // 2)
        end = min(time_steps, center + window_size // 2)
        
        if end - start < fs:  # Need at least 1 second of data
            continue
        
        window_x = combined_accel_x[start:end]
        window_y = combined_accel_y[start:end]
        
        # Compute PSD for this window
        freqs_x, psd_x = signal.welch(window_x, fs, nperseg=min(128, len(window_x)))
        freqs_y, psd_y = signal.welch(window_y, fs, nperseg=min(128, len(window_y)))
        
        # Average PSD across axes
        psd = (psd_x + psd_y) / 2
        
        # Power in each band
        rest_power = np.sum(psd[(freqs_x >= 0) & (freqs_x < 1)])
        postural_power = np.sum(psd[(freqs_x >= 1) & (freqs_x < 4)])
        kinetic_power = np.sum(psd[(freqs_x >= 4) & (freqs_x < 10)])
        
        total_power = rest_power + postural_power + kinetic_power + 1e-8
        
        # Softmax-like probabilities (not hard one-hot)
        rest_prob = rest_power / total_power
        postural_prob = postural_power / total_power
        kinetic_prob = kinetic_power / total_power
        
        # Fill the window with these probabilities
        context[start:end, 0] = rest_prob
        context[start:end, 1] = postural_prob
        context[start:end, 2] = kinetic_prob
    
    return context

def data_augmentation(window_data, augmentation_type='noise'):
    """Apply data augmentation to improve generalization"""
    augmented = window_data.copy()
    
    if augmentation_type == 'noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, augmented.shape)
        augmented += noise
    
    elif augmentation_type == 'time_warp':
        # Slight time warping
        from scipy.interpolate import interp1d
        time_steps = augmented.shape[0]
        orig_time = np.arange(time_steps)
        new_time = np.arange(time_steps) + np.random.uniform(-0.5, 0.5, time_steps)
        new_time = np.clip(new_time, 0, time_steps-1)
        
        for f in range(augmented.shape[1]):
            interpolator = interp1d(orig_time, augmented[:, f], kind='cubic', 
                                    bounds_error=False, fill_value='extrapolate')
            augmented[:, f] = interpolator(new_time)
    
    elif augmentation_type == 'amplitude_scale':
        # Random amplitude scaling
        scale = np.random.uniform(0.9, 1.1)
        augmented *= scale
    
    return augmented