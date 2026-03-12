import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from scipy import signal
from encoding_utils import (
    frequency_aware_encoding,
    compute_asymmetry_features,
    improved_context_encoding
)
import pandas as pd

class TIMTremorDataset(Dataset):
    """
    Dataset loader for Parkinson's tremor CSV files
    Loads ALL patient_*.csv files from directory (not limited to 20)
    """
    def __init__(self, data_dir, window_size=128, step_size=64, 
                 use_improved_encoding=True, augment=False, fs=100):
        """
        Args:
            data_dir: Directory containing patient_*.csv files
            window_size: Number of time steps per window
            step_size: Step size for sliding window (default 50% overlap)
            use_improved_encoding: If True, use frequency-aware encoding
            augment: If True, apply data augmentation
            fs: Sampling frequency (Hz)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.use_improved_encoding = use_improved_encoding
        self.augment = augment
        self.fs = fs
        
        # Find ALL patient CSV files (not limited to 20)
        self.csv_files = sorted(list(self.data_dir.glob("patient_*.csv")))
        
        if len(self.csv_files) == 0:
            raise FileNotFoundError(
                f"No patient CSV files found in {data_dir}\n"
                f"Expected files like: patient_001.csv, patient_002.csv, etc."
            )
        
        print(f"📁 Found {len(self.csv_files)} patient CSV files")
        
        # Load all data and create windows
        self.windows = []
        self.labels = []
        self.file_sources = []  # Track which file each window came from
        
        for csv_file in self.csv_files:
            try:
                # Load IMU data
                df = pd.read_csv(csv_file)
                
                # Verify expected columns exist
                expected_cols = ['Accel_X_Left', 'Accel_Y_Left', 'Accel_Z_Left',
                               'Accel_X_Right', 'Accel_Y_Right', 'Accel_Z_Right']
                
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    print(f"⚠️  Warning: {csv_file.name} missing columns: {missing_cols}")
                    continue
                
                data = df[expected_cols].values.astype(np.float32)
                
                # Calculate tremor severity from IMU data
                tremor_left = np.std(df['Accel_X_Left'].values)
                tremor_right = np.std(df['Accel_X_Right'].values)
                avg_tremor = (tremor_left + tremor_right) / 2
                
                # Heuristic for UPDRS score (0-3)
                if avg_tremor < 0.25:
                    patient_label = 0  # No tremor
                elif avg_tremor < 0.45:
                    patient_label = 1  # Slight tremor
                elif avg_tremor < 0.7:
                    patient_label = 2  # Mild tremor
                else:
                    patient_label = 3  # Moderate tremor
                
                # Create sliding windows
                num_windows = (len(data) - window_size) // step_size + 1
                
                for i in range(num_windows):
                    start_idx = i * step_size
                    end_idx = start_idx + window_size
                    window_data = data[start_idx:end_idx]
                    self.windows.append(window_data)
                    self.labels.append(patient_label)
                    self.file_sources.append(csv_file.name)
                
                print(f"  ✅ {csv_file.name}: {num_windows} windows (UPDRS={patient_label})")
                
            except Exception as e:
                print(f"  ❌ Error loading {csv_file.name}: {e}")
                continue
        
        if len(self.windows) == 0:
            raise ValueError("No valid windows loaded from CSV files!")
        
        print(f"\n✅ Created {len(self.windows)} total windows from {len(self.csv_files)} patients")
        
        # Print label distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\n📊 Label distribution:")
        for label, count in zip(unique, counts):
            print(f"   UPDRS {label}: {count} windows ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.windows)
    
    def _split_left_right(self, window_data):
        """Split window data into left and right hand data"""
        left_data = window_data[:, :3]   # First 3 columns = Left (X, Y, Z)
        right_data = window_data[:, 3:6] # Next 3 columns = Right (X, Y, Z)
        return left_data, right_data
    
    def __getitem__(self, idx):
        # Get window data and label
        window_data = self.windows[idx]
        label = self.labels[idx]
        
        # Split into left and right
        left_data, right_data = self._split_left_right(window_data)
        
        # Normalize
        left_mean = np.mean(left_data, axis=0, keepdims=True)
        left_std = np.std(left_data, axis=0, keepdims=True) + 1e-8
        right_mean = np.mean(right_data, axis=0, keepdims=True)
        right_std = np.std(right_data, axis=0, keepdims=True) + 1e-8
        
        left_data = (left_data - left_mean) / left_std
        right_data = (right_data - right_mean) / right_std
        
        # Apply augmentation
        if self.augment:
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, left_data.shape)
                left_data += noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.02, right_data.shape)
                right_data += noise
        
        if self.use_improved_encoding:
            # === IMPROVED ENCODING PIPELINE ===
            
            # 1. Frequency-aware spike encoding (tremor band 4-6 Hz)
            left_spikes = frequency_aware_encoding(
                left_data, self.fs, tremor_band=(4, 6), gain=8
            )
            right_spikes = frequency_aware_encoding(
                right_data, self.fs, tremor_band=(4, 6), gain=8
            )
            
            # 2. Compute asymmetry features
            asymmetry_features = compute_asymmetry_features(left_data, right_data, self.fs)
            asymmetry_spikes = torch.tensor(asymmetry_features, dtype=torch.float32)
            
            # 3. Context encoding
            context_features = improved_context_encoding(left_data, right_data, self.fs)
            context_spikes = torch.tensor(context_features, dtype=torch.float32)
            
            # 4. Concatenate asymmetry features
            left_spikes = torch.cat([left_spikes, asymmetry_spikes], dim=1)  # [128, 6]
            right_spikes = torch.cat([right_spikes, asymmetry_spikes], dim=1)  # [128, 6]
            
        else:
            # Fallback: Simple threshold encoding
            left_spikes = []
            right_spikes = []
            context_spikes = []
            
            for t in range(self.window_size):
                left_spike = (torch.sigmoid(torch.tensor(left_data[t])) > 0.5).float()
                right_spike = (torch.sigmoid(torch.tensor(right_data[t])) > 0.5).float()
                context_spike = torch.zeros(3)
                
                left_spikes.append(left_spike)
                right_spikes.append(right_spike)
                context_spikes.append(context_spike)
            
            left_spikes = torch.stack(left_spikes)
            right_spikes = torch.stack(right_spikes)
            context_spikes = torch.stack(context_spikes)
        
        return left_spikes, right_spikes, context_spikes, torch.tensor(label, dtype=torch.long)