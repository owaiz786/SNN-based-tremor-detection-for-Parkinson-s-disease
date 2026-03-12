import numpy as np
import pandas as pd
from pathlib import Path

def generate_patient_data(patient_id, tremor_severity, n_samples=6000, fs=100):
    """Generate IMU data for a patient with specific tremor severity"""
    
    t = np.arange(n_samples) / fs
    
    # Base tremor frequency (4-6 Hz)
    tremor_freq = 5 + np.random.uniform(-1, 1)
    
    # Scale tremor amplitude based on severity
    if tremor_severity == 0:  # No tremor
        tremor_amp_left = np.random.uniform(0, 0.2)
        tremor_amp_right = np.random.uniform(0, 0.2)
    elif tremor_severity == 1:  # Slight
        tremor_amp_left = np.random.uniform(0.3, 0.5)
        tremor_amp_right = np.random.uniform(0.3, 0.5)
    elif tremor_severity == 2:  # Mild
        tremor_amp_left = np.random.uniform(0.6, 0.9)
        tremor_amp_right = np.random.uniform(0.5, 0.8)
    else:  # Moderate (3)
        tremor_amp_left = np.random.uniform(1.0, 1.5)
        tremor_amp_right = np.random.uniform(0.9, 1.3)
    
    # Generate tremor signals
    tremor_left = tremor_amp_left * np.sin(2 * np.pi * tremor_freq * t) + np.random.normal(0, 0.2, n_samples)
    tremor_right = tremor_amp_right * np.sin(2 * np.pi * (tremor_freq + np.random.uniform(-0.5, 0.5)) * t) + np.random.normal(0, 0.15, n_samples)
    
    # Add bradykinesia patterns
    brady_pattern = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.normal(0, 1, n_samples)
    
    # Generate IMU data
    data = pd.DataFrame({
        'Accel_X_Left': tremor_left + 0.1 * brady_pattern,
        'Accel_Y_Left': 0.5 * tremor_left + np.random.normal(0, 0.3, n_samples),
        'Accel_Z_Left': np.random.normal(9.8, 0.5, n_samples),
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
    
    return data

def main():
    data_dir = Path('real_pd_data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate 20 patients with various tremor severities
    n_patients = 20
    patients_per_severity = n_patients // 4
    
    metadata = []
    
    for severity in range(4):
        for i in range(patients_per_severity):
            patient_id = f"{severity}{i+1:02d}"
            data = generate_patient_data(patient_id, severity)
            
            # Save patient data
            data.to_csv(data_dir / f'patient_{patient_id}.csv', index=False)
            
            # Add to metadata
            metadata.append({
                'patient_id': patient_id,
                'tremor_severity': severity,
                'age': np.random.randint(50, 85),
                'gender': np.random.choice(['M', 'F']),
                'sampling_rate': 100,
                'duration_seconds': 60
            })
            
            print(f"Generated patient {patient_id} with severity {severity}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(data_dir / 'metadata.csv', index=False)
    print(f"\n✅ Generated {n_patients} patient files")
    print("\nSeverity distribution:")
    print(metadata_df['tremor_severity'].value_counts().sort_index())

if __name__ == "__main__":
    main()