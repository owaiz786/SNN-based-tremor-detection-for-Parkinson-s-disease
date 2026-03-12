import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class TIMTremorSNN(nn.Module):
    def __init__(self, num_classes=4, use_asymmetry_features=True):
        """
        Adapted for improved encoding with asymmetry features
        
        Args:
            num_classes: Number of UPDRS classes (default 4)
            use_asymmetry_features: If True, expect 6 input features (3 accel + 3 asymmetry)
        """
        super().__init__()
        
        # Surrogate gradient for backpropagation through non-differentiable spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.95  # Increased from 0.9 for better temporal integration
        
        # Input feature dimension
        input_dim = 6 if use_asymmetry_features else 3
        
        # -------------------------------------------------------------------
        # PATHWAY 1: Left Arm Lobe
        # -------------------------------------------------------------------
        self.fc_left1 = nn.Linear(input_dim, 32)  # Increased from 16 to 32
        self.lif_left1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_left2 = nn.Linear(32, 32)
        self.lif_left2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------------------------------------------------------------------
        # PATHWAY 2: Right Arm Lobe
        # -------------------------------------------------------------------
        self.fc_right1 = nn.Linear(input_dim, 32)
        self.lif_right1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_right2 = nn.Linear(32, 32)
        self.lif_right2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
 
        # -------------------------------------------------------------------
        # PATHWAY 3: Context Lobe (3 inputs: rest, postural, kinetic probabilities)
        # -------------------------------------------------------------------
        self.fc_ctx = nn.Linear(3, 16)  # Increased from 8 to 16
        self.lif_ctx = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------------------------------------------------------------------
        # FUSION LOBE: Left(32) + Right(32) + Context(16) = 80 input features
        # -------------------------------------------------------------------
        self.fc_fusion = nn.Linear(80, 64)  # Increased capacity
        self.lif_fusion = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Additional fusion layer for better integration
        self.fc_fusion2 = nn.Linear(64, 48)
        self.lif_fusion2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Final Classification Layer
        self.fc_out = nn.Linear(48, num_classes) 
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, left_spikes, right_spikes, context_spikes):
        """
        Input shapes expected: [time_steps, batch_size, features]
        left_spikes: [time_steps, batch_size, 6] (with asymmetry)
        right_spikes: [time_steps, batch_size, 6] (with asymmetry)
        context_spikes: [time_steps, batch_size, 3]
        """
        # Initialize membrane potentials
        mem_l1 = self.lif_left1.init_leaky()
        mem_l2 = self.lif_left2.init_leaky()
        
        mem_r1 = self.lif_right1.init_leaky()
        mem_r2 = self.lif_right2.init_leaky()
        
        mem_c = self.lif_ctx.init_leaky()
        
        mem_f = self.lif_fusion.init_leaky()
        mem_f2 = self.lif_fusion2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_out_rec = []
        
        # Iterate through time steps
        num_steps = left_spikes.size(0)
        for step in range(num_steps):
            
            # Process Left Arm
            cur_l1 = self.fc_left1(left_spikes[step])
            spk_l1, mem_l1 = self.lif_left1(cur_l1, mem_l1)
            cur_l2 = self.fc_left2(spk_l1)
            spk_l2, mem_l2 = self.lif_left2(cur_l2, mem_l2)
            
            # Process Right Arm
            cur_r1 = self.fc_right1(right_spikes[step])
            spk_r1, mem_r1 = self.lif_right1(cur_r1, mem_r1)
            cur_r2 = self.fc_right2(spk_r1)
            spk_r2, mem_r2 = self.lif_right2(cur_r2, mem_r2)
            
            # Process Context
            cur_c = self.fc_ctx(context_spikes[step])
            spk_c, mem_c = self.lif_ctx(cur_c, mem_c)
            
            # Fuse pathways
            fused_spikes = torch.cat([spk_l2, spk_r2, spk_c], dim=1)
            
            cur_f = self.fc_fusion(fused_spikes)
            spk_f, mem_f = self.lif_fusion(cur_f, mem_f)
            
            # Second fusion layer for better integration
            cur_f2 = self.fc_fusion2(spk_f)
            spk_f2, mem_f2 = self.lif_fusion2(cur_f2, mem_f2)
            
            # Final prediction
            cur_out = self.fc_out(spk_f2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            spk_out_rec.append(spk_out)
        
        return torch.stack(spk_out_rec, dim=0)