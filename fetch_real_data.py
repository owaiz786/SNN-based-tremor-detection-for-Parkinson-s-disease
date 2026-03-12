import numpy as np
import pandas as pd
import os
from pathlib import Path

def download_pd_data():
    """Generate realistic synthetic Parkinson's IMU data"""
    
    # Create data directory
    data_dir = Path('real_pd_data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic but realistic IMU data for Parkinson's patients
    n_samples = 6000  # 60 seconds at 100Hz
    fs = 100  # Sampling frequency
    
    # Time vector
    t = np.arange(n_samples) / fs
    
    # Generate realistic tremor signals (4-6 Hz Parkinsonian tremor)
    tremor_freq = 5  # Hz
    tremor_left = 0.8 * np.sin(2 * np.pi * tremor_freq * t) + np.random.normal(0, 0.2, n_samples)
    tremor_right = 0.6 * np.sin(2 * np.pi * (tremor_freq + 0.5) * t) + np.random.normal(0, 0.15, n_samples)
    
    # Add bradykinesia (slowness of movement) patterns
    brady_pattern = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.normal(0, 1, n_samples)
    
    # Generate synthetic IMU data
    data = pd.DataFrame({
        'Accel_X_Left': tremor_left + 0.1 * brady_pattern,
        'Accel_Y_Left': 0.5 * tremor_left + np.random.normal(0, 0.3, n_samples),
        'Accel_Z_Left': np.random.normal(9.8, 0.5, n_samples),  # Gravity in Z
        'Accel_X_Right': tremor_right + 0.1 * brady_pattern,
        'Accel_Y_Right': 0.5 * tremor_right + np.random.normal(0, 0.25, n_samples),
        'Accel_Z_Right': np.random.normal(9.8, 0.5, n_samples),
        'Gyro_X_Left': 20 * tremor_left + np.random.normal(0, 5, n_samples),
        'Gyro_Y_Left': 15 * tremor_left + np.random.normal(0, 4, n_samples),
        'Gyro_Z_Left': np.random.normal(0, 3, n_samples),
        'Gyro_X_Right': 18 * tremor_right + np.random.normal(0, 4.5, n_samples),
        'Gyro_Y_Right': 12 * tremor_right + np.random.normal(0, 3.5, n_samples),
        'Gyro_Z_Right': np.random.normal(0, 2.5, n_samples),
    })
    
    # Add some realistic drift and offset
    drift = 0.001 * np.arange(n_samples)[:, np.newaxis]
    data += drift * np.random.uniform(-1, 1, data.shape[1])
    
    # Save to CSV
    data.to_csv(data_dir / 'patient_001.csv', index=False)
    print(f"Generated synthetic data saved to {data_dir / 'patient_001.csv'}")
    
    # Create a simple metadata file
    metadata = pd.DataFrame({
        'patient_id': ['001'],
        'condition': ['Parkinson\'s Disease'],
        'age': [65],
        'gender': ['M'],
        'updrs_score': [45],
        'medication': ['Levodopa'],
        'sampling_rate': [fs],
        'duration_seconds': [n_samples / fs]
    })
    metadata.to_csv(data_dir / 'metadata.csv', index=False)
    print(f"Metadata saved to {data_dir / 'metadata.csv'}")
    
    return data

if __name__ == "__main__":
    print("Generating Realistic Parkinson's IMU Data...")
    download_pd_data()
    print("Done!")