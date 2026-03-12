import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TremorClinicalDataset(Dataset):
    def __init__(self, csv_file, window_size=200, step_size=100):
        """
        Args:
            csv_file: Path to the CSV file with clinical data
            window_size: Number of time steps per window
            step_size: Step size for sliding window
        """
        self.df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.step_size = step_size
        
        # Check which column name exists for the UPDRS score
        if 'Tremor_Severity_UPDRS' in self.df.columns:
            self.label_col = 'Tremor_Severity_UPDRS'
        elif 'updrs_tremor_score' in self.df.columns:
            self.label_col = 'updrs_tremor_score'
        else:
            # Try to find any column that might contain UPDRS/score information
            possible_columns = [col for col in self.df.columns if 'updrs' in col.lower() or 'tremor' in col.lower() or 'score' in col.lower()]
            if possible_columns:
                self.label_col = possible_columns[0]
                print(f"⚠️  Using '{self.label_col}' as the label column")
            else:
                raise KeyError("No suitable UPDRS score column found in the CSV file")
        
        # Separate features and labels
        feature_columns = [col for col in self.df.columns if col != self.label_col]
        self.features = self.df[feature_columns].values
        self.labels = self.df[self.label_col].values
        
        # Calculate number of windows
        self.num_windows = max(0, (len(self.features) - window_size) // step_size + 1)
        
        print(f"📊 Dataset loaded: {len(self.features)} samples, {self.num_windows} windows")
        print(f"   Features: {len(feature_columns)}")
        print(f"   Label column: {self.label_col}")
        print(f"   Label distribution:\n{self.df[self.label_col].value_counts().sort_index()}")
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        # Calculate window indices
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        
        # Get window of features
        window_features = self.features[start_idx:end_idx]
        
        # Get label for this window (use majority vote or first value)
        window_labels = self.labels[start_idx:end_idx]
        label = int(np.bincount(window_labels).argmax())  # Majority vote
        
        # For now, we need to convert features to spike trains
        # This is a placeholder - you'll need proper spike encoding
        # Let's create simple spike trains by thresholding
        num_features = window_features.shape[1]
        
        # Normalize features
        window_features = (window_features - window_features.mean()) / (window_features.std() + 1e-8)
        
        # Create spike trains (simplified rate coding)
        # For each feature, generate spikes based on feature value
        left_spikes = []
        right_spikes = []
        ctx_spikes = []
        
        # Split features into left, right, and contextual
        # This is an assumption - adjust based on your actual feature columns
        n_left = num_features // 3
        n_right = num_features // 3
        n_ctx = num_features - n_left - n_right
        
        for t in range(self.window_size):
            # Rate-based encoding: spike probability proportional to feature value
            left_probs = torch.sigmoid(torch.tensor(window_features[t, :n_left]))
            right_probs = torch.sigmoid(torch.tensor(window_features[t, n_left:n_left+n_right]))
            ctx_probs = torch.sigmoid(torch.tensor(window_features[t, n_left+n_right:]))
            
            # Generate spikes (Bernoulli sampling)
            left_spike = torch.bernoulli(left_probs).float()
            right_spike = torch.bernoulli(right_probs).float()
            ctx_spike = torch.bernoulli(ctx_probs).float()
            
            left_spikes.append(left_spike)
            right_spikes.append(right_spike)
            ctx_spikes.append(ctx_spike)
        
        # Stack into tensors [window_size, num_features]
        left_spikes = torch.stack(left_spikes)
        right_spikes = torch.stack(right_spikes)
        ctx_spikes = torch.stack(ctx_spikes)
        
        return left_spikes, right_spikes, ctx_spikes, torch.tensor(label, dtype=torch.long)