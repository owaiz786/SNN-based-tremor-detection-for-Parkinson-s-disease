import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from encoding_utils import (
    frequency_aware_encoding,
    compute_asymmetry_features,
    improved_context_encoding,
    bandpass_filter,
    data_augmentation
)

class IMUTremorDataset(Dataset):
    def __init__(self, data_dir, window_size=200, step_size=100, use_improved_encoding=True, 
                 augment=False, fs=100):
        """
        Args:
            data_dir: Directory containing the IMU CSV files
            window_size: Number of time steps per window
            step_size: Step size for sliding window
            use_improved_encoding: If True, use frequency-aware encoding (RECOMMENDED)
            augment: If True, apply data augmentation
            fs: Sampling frequency (Hz)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.use_improved_encoding = use_improved_encoding
        self.augment = augment
        self.fs = fs
        
        # Find all patient CSV files
        self.csv_files = list(self.data_dir.glob("patient_*.csv"))

        if len(self.csv_files) == 0:
            raise FileNotFoundError(f"No patient CSV files found in {data_dir}")

        print(f"📁 Found {len(self.csv_files)} patient data files")

        # Load all data and create windows
        self.windows = []
        self.labels = []
        self.asymmetry_scores = []  # Track for analysis
        self.patient_ids = []  # Track for patient-level validation

        for csv_file in self.csv_files:
            # Load IMU data
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            required_cols = ['Accel_X_Left', 'Accel_Y_Left', 'Accel_Z_Left',
                            'Accel_X_Right', 'Accel_Y_Right', 'Accel_Z_Right']
            
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️  Skipping {csv_file}: Missing required columns")
                continue
                
            data = df[required_cols].values.astype(np.float32)

            # Calculate tremor severity from IMU data (IMPROVED)
            # Use spectral power in 4-6 Hz band instead of simple std
            left_powers = self._compute_tremor_power(df['Accel_X_Left'].values)
            right_powers = self._compute_tremor_power(df['Accel_X_Right'].values)
            
            avg_tremor_power = (left_powers + right_powers) / 2

            # Heuristic for UPDRS score (0-3) - IMPROVED thresholds based on empirical data
            # These thresholds should be calibrated on your specific dataset
            if avg_tremor_power < 0.1:
                patient_label = 0  # No tremor
            elif avg_tremor_power < 0.25:
                patient_label = 1  # Slight tremor
            elif avg_tremor_power < 0.45:
                patient_label = 2  # Mild tremor
            else:
                patient_label = 3  # Moderate tremor

            # Compute asymmetry score for this patient
            asymmetry = np.abs(left_powers - right_powers) / (left_powers + right_powers + 1e-8)
            self.asymmetry_scores.append(asymmetry)

            print(
                f"  Patient {csv_file.stem}: Tremor power={avg_tremor_power:.3f}, "
                f"Asymmetry={asymmetry:.3f} → UPDRS={patient_label}"
            )

            # Create sliding windows
            num_windows = (len(data) - window_size) // step_size + 1

            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                window_data = data[start_idx:end_idx]
                
                # Skip windows with too much noise (optional)
                if np.std(window_data) < 0.01:
                    continue
                    
                self.windows.append(window_data)
                self.labels.append(patient_label)
                self.patient_ids.append(csv_file.stem)

        print(f"✅ Created {len(self.windows)} windows from {len(self.csv_files)} patients")

        # Print label distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("   Label distribution:")
        for label, count in zip(unique, counts):
            print(f"     UPDRS {label}: {count} windows ({count/len(self.labels)*100:.1f}%)")

    def _compute_tremor_power(self, data):
        """Compute spectral power in 4-6 Hz Parkinson's tremor band"""
        if len(data) < 10:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=min(128, len(data)))
        tremor_mask = (freqs >= 4) & (freqs <= 6)
        if np.any(tremor_mask):
            return np.sum(psd[tremor_mask])
        return 0

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Get window data and label
        window_data = self.windows[idx].copy()
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        # Apply augmentation if enabled (for training)
        if self.augment:
            aug_type = np.random.choice(['noise', 'time_warp', 'amplitude_scale', 'none'], 
                                        p=[0.3, 0.2, 0.2, 0.3])
            if aug_type != 'none':
                window_data = data_augmentation(window_data, aug_type)

        # Split into left and right (3 accel channels each)
        left_data = window_data[:, :3]  # X, Y, Z for left
        right_data = window_data[:, 3:6]  # X, Y, Z for right

        # Normalize (z-score per channel) with robust scaling
        left_mean = np.mean(left_data, axis=0, keepdims=True)
        left_std = np.std(left_data, axis=0, keepdims=True) + 1e-8
        right_mean = np.mean(right_data, axis=0, keepdims=True)
        right_std = np.std(right_data, axis=0, keepdims=True) + 1e-8
        
        left_data = (left_data - left_mean) / left_std
        right_data = (right_data - right_mean) / right_std

        if self.use_improved_encoding:
            # === IMPROVED ENCODING PIPELINE ===
            
            # 1. Frequency-aware spike encoding for left/right
            left_spikes = frequency_aware_encoding(left_data, self.fs, tremor_band=(4, 6), gain=8)
            right_spikes = frequency_aware_encoding(right_data, self.fs, tremor_band=(4, 6), gain=8)
            
            # 2. Compute explicit asymmetry features (CRITICAL ADDITION)
            asymmetry_features = compute_asymmetry_features(left_data, right_data, self.fs)
            asymmetry_spikes = torch.tensor(asymmetry_features, dtype=torch.float32)
            
            # 3. Improved context encoding (full-window PSD)
            context_features = improved_context_encoding(left_data, right_data, self.fs)
            context_spikes = torch.tensor(context_features, dtype=torch.float32)
            
            # 4. Create multi-channel input for model
            # Concatenate asymmetry to both left and right (so model sees it)
            left_spikes = torch.cat([left_spikes, asymmetry_spikes], dim=1)  # [time, 3+3=6]
            right_spikes = torch.cat([right_spikes, asymmetry_spikes], dim=1)  # [time, 3+3=6]
            
        else:
            # === FALLBACK: Original simple encoding ===
            left_spikes = []
            right_spikes = []
            ctx_spikes = []

            for t in range(self.window_size):
                left_spike = (torch.sigmoid(torch.tensor(left_data[t])) > 0.5).float()
                right_spike = (torch.sigmoid(torch.tensor(right_data[t])) > 0.5).float()
                ctx_spike = torch.zeros(3)
                
                left_spikes.append(left_spike)
                right_spikes.append(right_spike)
                ctx_spikes.append(ctx_spike)

            left_spikes = torch.stack(left_spikes)
            right_spikes = torch.stack(right_spikes)
            ctx_spikes = torch.stack(ctx_spikes)

        return left_spikes, right_spikes, context_spikes, torch.tensor(label, dtype=torch.long)