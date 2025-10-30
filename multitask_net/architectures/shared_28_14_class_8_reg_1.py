"""
Multitask Spiking Neural Network Architecture: Shared-28-14 + Class-8 + Reg-1

This module implements a shared SNN with task-specific branches:
- Shared layers: 28 → 14 LIF neurons (process input from both wine and ethanol data)
- Classification branch: 14 → 8 → 3 LIF neurons (Wine quality prediction)
- Regression branch: 14 → 1 LIF neuron (Ethanol concentration prediction)

Architecture Name: shared_28_14_class_8_reg_1
"""

import torch
import torch.nn as nn
import snntorch as snn


class MultitaskSNN(nn.Module):
    """
    Multitask Spiking Neural Network with shared and task-specific layers.
    
    Architecture (based on Figure 17):
    - Shared Layers:
        * Input: 57 features (Array #1, Array #8, Analyte on/off)
        * Hidden 1: 28 LIF neurons (shared)
        * Hidden 2: 14 LIF neurons (shared)
    
    - Classification Branch (Analyte Classification):
        * Hidden 3: 8 LIF neurons
        * Output: 3 neurons (HQ, LQ, AQ wine quality classes)
    
    - Regression Branch (Concentration Estimation):
        * Output: 1 neuron (continuous ethanol concentration)
    """
    
    # Architecture identifier for model saving
    ARCHITECTURE_NAME = "shared_28_14_class_8_reg_1"
    TRAINING_SCRIPT = "train_two_shared_layers.py"
    
    def __init__(
        self,
        input_size=32,
        shared_hidden1=28,
        shared_hidden2=14,
        classification_hidden=8,
        num_classes=3,
        beta=0.9,
        spike_grad=None
    ):
        """
        Args:
            input_size: Number of input features
            shared_hidden1: Size of first shared hidden layer (28 from diagram)
            shared_hidden2: Size of second shared hidden layer (14 from diagram)
            classification_hidden: Size of classification branch hidden layer (8 from diagram)
            num_classes: Number of classification classes (3 for wine quality)
            beta: Membrane potential decay rate (0 < beta < 1)
            spike_grad: Surrogate gradient function for backpropagation
        """
        super(MultitaskSNN, self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.shared_hidden1 = shared_hidden1
        self.shared_hidden2 = shared_hidden2
        self.classification_hidden = classification_hidden
        self.num_classes = num_classes
        
        # ====================================================================
        # SHARED LAYERS (Purple in diagram)
        # ====================================================================
        self.fc_shared1 = nn.Linear(input_size, shared_hidden1)
        self.lif_shared1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_shared2 = nn.Linear(shared_hidden1, shared_hidden2)
        self.lif_shared2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # ====================================================================
        # CLASSIFICATION BRANCH (Red in diagram - Analyte Classification)
        # ====================================================================
        self.fc_class1 = nn.Linear(shared_hidden2, classification_hidden)
        self.lif_class1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_class_out = nn.Linear(classification_hidden, num_classes)
        self.lif_class_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # ====================================================================
        # REGRESSION BRANCH (Teal in diagram - Concentration Estimation)
        # ====================================================================
        self.fc_reg_out = nn.Linear(shared_hidden2, 1)
        self.lif_reg_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        """
        Forward pass through the multitask SNN.
        
        Args:
            x: Input spike train of shape (time_steps, batch_size, input_size)
            
        Returns:
            Dictionary containing:
            - classification: (mem_rec, spk_rec) for classification output
            - regression: (mem_rec, spk_rec) for regression output
            - shared_spikes: (spk_rec1, spk_rec2) from shared layers
        """
        # Initialize membrane potentials for all layers
        mem_shared1 = self.lif_shared1.init_leaky()
        mem_shared2 = self.lif_shared2.init_leaky()
        mem_class1 = self.lif_class1.init_leaky()
        mem_class_out = self.lif_class_out.init_leaky()
        mem_reg_out = self.lif_reg_out.init_leaky()
        
        # Recording lists
        spk_shared1_rec = []
        spk_shared2_rec = []
        spk_class1_rec = []
        mem_class_out_rec = []
        mem_reg_out_rec = []
        
        # Process each time step
        num_steps = x.size(0)
        for step in range(num_steps):
            x_step = x[step]  # (batch_size, input_size)
            
            # ================================================================
            # SHARED LAYERS
            # ================================================================
            # Layer 1: Input → Shared Hidden 1
            cur_shared1 = self.fc_shared1(x_step)
            spk_shared1, mem_shared1 = self.lif_shared1(cur_shared1, mem_shared1)
            
            # Layer 2: Shared Hidden 1 → Shared Hidden 2
            cur_shared2 = self.fc_shared2(spk_shared1)
            spk_shared2, mem_shared2 = self.lif_shared2(cur_shared2, mem_shared2)
            
            # ================================================================
            # CLASSIFICATION BRANCH
            # ================================================================
            # Classification Hidden Layer
            cur_class1 = self.fc_class1(spk_shared2)
            spk_class1, mem_class1 = self.lif_class1(cur_class1, mem_class1)
            
            # Classification Output
            cur_class_out = self.fc_class_out(spk_class1)
            spk_class_out, mem_class_out = self.lif_class_out(cur_class_out, mem_class_out)
            
            # ================================================================
            # REGRESSION BRANCH
            # ================================================================
            # Regression Output (directly from shared layer 2)
            cur_reg_out = self.fc_reg_out(spk_shared2)
            spk_reg_out, mem_reg_out = self.lif_reg_out(cur_reg_out, mem_reg_out)
            
            # Record activity
            spk_shared1_rec.append(spk_shared1)
            spk_shared2_rec.append(spk_shared2)
            spk_class1_rec.append(spk_class1)
            mem_class_out_rec.append(mem_class_out)
            mem_reg_out_rec.append(mem_reg_out)
        
        # Stack recordings
        spk_shared1_rec = torch.stack(spk_shared1_rec)
        spk_shared2_rec = torch.stack(spk_shared2_rec)
        spk_class1_rec = torch.stack(spk_class1_rec)
        mem_class_out_rec = torch.stack(mem_class_out_rec)
        mem_reg_out_rec = torch.stack(mem_reg_out_rec)
        
        return {
            'classification': {
                'mem_rec': mem_class_out_rec,
                'spk_rec': spk_class1_rec
            },
            'regression': {
                'mem_rec': mem_reg_out_rec
            },
            'shared': {
                'spk_rec1': spk_shared1_rec,
                'spk_rec2': spk_shared2_rec
            }
        }
    
    def get_num_parameters(self):
        """Calculate total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_architecture_summary(self):
        """Return a string summary of the architecture."""
        total, trainable = self.get_num_parameters()
        
        summary = f"""
{'='*80}
MULTITASK SNN ARCHITECTURE: {self.ARCHITECTURE_NAME}
{'='*80}

SHARED LAYERS (Purple - Common Processing):
  Input Size:              {self.input_size}
  Shared Hidden Layer 1:   {self.shared_hidden1} LIF neurons
  Shared Hidden Layer 2:   {self.shared_hidden2} LIF neurons

CLASSIFICATION BRANCH (Red - Analyte Classification):
  Classification Hidden:   {self.classification_hidden} LIF neurons
  Output:                  {self.num_classes} classes (Wine Quality)

REGRESSION BRANCH (Teal - Concentration Estimation):
  Output:                  1 neuron (Ethanol Concentration)

PARAMETERS:
  Total:                   {total:,}
  Trainable:               {trainable:,}
{'='*80}
"""
        return summary
