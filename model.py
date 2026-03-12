import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class MultiPathwayBilateralSNN(nn.Module):
    def __init__(self, num_classes=4):
        # num_classes = 4 (UPDRS Severity: 0=None, 1=Slight, 2=Mild, 3=Moderate/Severe)
        super().__init__()
        
        # Surrogate gradient for backpropagation through non-differentiable spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.9  # Membrane potential decay rate
        
        # -------------------------------------------------------------------
        # PATHWAY 1: Left Arm Lobe (6 Inputs: Pos/Neg Spikes for X, Y, Z)
        # -------------------------------------------------------------------
        self.fc_left1 = nn.Linear(6, 16)
        self.lif_left1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_left2 = nn.Linear(16, 16)
        self.lif_left2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------------------------------------------------------------------
        # PATHWAY 2: Right Arm Lobe (6 Inputs: Pos/Neg Spikes for X, Y, Z)
        # -------------------------------------------------------------------
        self.fc_right1 = nn.Linear(6, 16)
        self.lif_right1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_right2 = nn.Linear(16, 16)
        self.lif_right2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------------------------------------------------------------------
        # PATHWAY 3: Context Lobe (3 Inputs: One-Hot for Rest, Posture, Kinetic)
        # -------------------------------------------------------------------
        self.fc_ctx = nn.Linear(3, 8)
        self.lif_ctx = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------------------------------------------------------------------
        # FUSION LOBE: Bringing Left, Right, and Context together
        # Left(16) + Right(16) + Context(8) = 40 input features
        # -------------------------------------------------------------------
        self.fc_fusion = nn.Linear(40, 32)
        self.lif_fusion = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Final Classification Layer (Predicts UPDRS Tremor Severity)
        self.fc_out = nn.Linear(32, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, left_spikes, right_spikes, context_spikes):
        """
        Input shapes expected: [time_steps, batch_size, features]
        """
        # 1. Initialize membrane potentials for all neurons
        mem_l1 = self.lif_left1.init_leaky()
        mem_l2 = self.lif_left2.init_leaky()
        
        mem_r1 = self.lif_right1.init_leaky()
        mem_r2 = self.lif_right2.init_leaky()
        
        mem_c = self.lif_ctx.init_leaky()
        
        mem_f = self.lif_fusion.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_out_rec =[]
        
        # 2. Iterate through the sliding window time steps
        num_steps = left_spikes.size(0)
        for step in range(num_steps):
            
            # --- Process Left Arm ---
            cur_l1 = self.fc_left1(left_spikes[step])
            spk_l1, mem_l1 = self.lif_left1(cur_l1, mem_l1)
            cur_l2 = self.fc_left2(spk_l1)
            spk_l2, mem_l2 = self.lif_left2(cur_l2, mem_l2)
            
            # --- Process Right Arm ---
            cur_r1 = self.fc_right1(right_spikes[step])
            spk_r1, mem_r1 = self.lif_right1(cur_r1, mem_r1)
            cur_r2 = self.fc_right2(spk_r1)
            spk_r2, mem_r2 = self.lif_right2(cur_r2, mem_r2)
            
            # --- Process Context ---
            cur_c = self.fc_ctx(context_spikes[step])
            spk_c, mem_c = self.lif_ctx(cur_c, mem_c)
            
            # --- FUSE THE PATHWAYS ---
            # Concatenate the spikes from Left, Right, and Context
            fused_spikes = torch.cat([spk_l2, spk_r2, spk_c], dim=1) # Shape:[batch_size, 40]
            
            cur_f = self.fc_fusion(fused_spikes)
            spk_f, mem_f = self.lif_fusion(cur_f, mem_f)
            
            # --- Final Prediction ---
            cur_out = self.fc_out(spk_f)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            # Record output spikes
            spk_out_rec.append(spk_out)
            
        # Return the recorded spikes across all time steps
        return torch.stack(spk_out_rec, dim=0)