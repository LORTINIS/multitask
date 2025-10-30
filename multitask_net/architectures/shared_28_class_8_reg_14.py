"""
Multitask Spiking Neural Network Architecture: Shared-28 + Class-8 + Reg-14

This module implements a shared SNN with task-specific branches matching the diagram:
- Shared layer: 28 LIF neurons (process input from both wine and ethanol data)
- Classification branch: 28 → 8 → 4 LIF neurons (Wine quality prediction - 4 outputs)
- Regression branch: 28 → 14 → 1 LIF neuron (Ethanol concentration prediction)

Architecture Name: shared_28_class_8_reg_14
"""

import torch
import torch.nn as nn
import snntorch as snn


class MultitaskSNN(nn.Module):
    """
    Multitask Spiking Neural Network with shared and task-specific layers.
    
    Architecture (based on the provided diagram):
    - Shared Layer:
        * Input: 57 features (Array #1, Array #8, Analyte on/off)
        * Hidden: 28 LIF neurons (shared between both tasks)
    
    - Classification Branch (Analyte Classification):
        * Hidden: 8 LIF neurons
        * Output: 4 neurons (Acetonitrile, DCM, Methanol, Toluene)
                  NOTE: Adapted to 3 neurons for wine quality (HQ, LQ, AQ)
    
    - Regression Branch (Concentration Estimation):
        * Hidden: 14 LIF neurons
        * Output: 1 neuron (continuous ethanol concentration)
    """
    
    # Architecture identifier for model saving
    ARCHITECTURE_NAME = "shared_28_class_8_reg_14"
    TRAINING_SCRIPT = "train_single_shared_layer.py"
    
    def __init__(
        self,
        input_size=32,
        shared_hidden=28,
        classification_hidden=8,
        regression_hidden=14,
        num_classes=3,
        beta=0.9,
        spike_grad=None
    ):
        """
        Args:
            input_size: Number of input features
            shared_hidden: Size of shared hidden layer (28 from diagram)
            classification_hidden: Size of classification branch hidden layer (8 from diagram)
            regression_hidden: Size of regression branch hidden layer (14 from diagram)
            num_classes: Number of classification classes (3 for wine quality)
            beta: Membrane potential decay rate (0 < beta < 1)
            spike_grad: Surrogate gradient function for backpropagation
        """
        super(MultitaskSNN, self).__init__()
        
        # Store architecture parameters
        self.input_size = input_size
        self.shared_hidden = shared_hidden
        self.classification_hidden = classification_hidden
        self.regression_hidden = regression_hidden
        self.num_classes = num_classes
        
        # ====================================================================
        # SHARED LAYER (Purple in diagram)
        # ====================================================================
        self.fc_shared = nn.Linear(input_size, shared_hidden)
        self.lif_shared = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # ====================================================================
        # CLASSIFICATION BRANCH (Red in diagram - Analyte Classification)
        # ====================================================================
        self.fc_class_hidden = nn.Linear(shared_hidden, classification_hidden)
        self.lif_class_hidden = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_class_out = nn.Linear(classification_hidden, num_classes)
        self.lif_class_out = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # ====================================================================
        # REGRESSION BRANCH (Teal in diagram - Concentration Estimation)
        # ====================================================================
        self.fc_reg_hidden = nn.Linear(shared_hidden, regression_hidden)
        self.lif_reg_hidden = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_reg_out = nn.Linear(regression_hidden, 1)
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
            - shared_spikes: spk_rec from shared layer
        """
        # Initialize membrane potentials for all layers
        mem_shared = self.lif_shared.init_leaky()
        mem_class_hidden = self.lif_class_hidden.init_leaky()
        mem_class_out = self.lif_class_out.init_leaky()
        mem_reg_hidden = self.lif_reg_hidden.init_leaky()
        mem_reg_out = self.lif_reg_out.init_leaky()
        
        # Recording lists
        spk_shared_rec = []
        spk_class_hidden_rec = []
        mem_class_out_rec = []
        spk_reg_hidden_rec = []
        mem_reg_out_rec = []
        
        # Process each time step
        num_steps = x.size(0)
        for step in range(num_steps):
            x_step = x[step]  # (batch_size, input_size)
            
            # ================================================================
            # SHARED LAYER
            # ================================================================
            cur_shared = self.fc_shared(x_step)
            spk_shared, mem_shared = self.lif_shared(cur_shared, mem_shared)
            
            # ================================================================
            # CLASSIFICATION BRANCH
            # ================================================================
            # Classification Hidden Layer
            cur_class_hidden = self.fc_class_hidden(spk_shared)
            spk_class_hidden, mem_class_hidden = self.lif_class_hidden(cur_class_hidden, mem_class_hidden)
            
            # Classification Output
            cur_class_out = self.fc_class_out(spk_class_hidden)
            spk_class_out, mem_class_out = self.lif_class_out(cur_class_out, mem_class_out)
            
            # ================================================================
            # REGRESSION BRANCH
            # ================================================================
            # Regression Hidden Layer
            cur_reg_hidden = self.fc_reg_hidden(spk_shared)
            spk_reg_hidden, mem_reg_hidden = self.lif_reg_hidden(cur_reg_hidden, mem_reg_hidden)
            
            # Regression Output
            cur_reg_out = self.fc_reg_out(spk_reg_hidden)
            spk_reg_out, mem_reg_out = self.lif_reg_out(cur_reg_out, mem_reg_out)
            
            # Record activity
            spk_shared_rec.append(spk_shared)
            spk_class_hidden_rec.append(spk_class_hidden)
            mem_class_out_rec.append(mem_class_out)
            spk_reg_hidden_rec.append(spk_reg_hidden)
            mem_reg_out_rec.append(mem_reg_out)
        
        # Stack recordings
        spk_shared_rec = torch.stack(spk_shared_rec)
        spk_class_hidden_rec = torch.stack(spk_class_hidden_rec)
        mem_class_out_rec = torch.stack(mem_class_out_rec)
        spk_reg_hidden_rec = torch.stack(spk_reg_hidden_rec)
        mem_reg_out_rec = torch.stack(mem_reg_out_rec)
        
        return {
            'classification': {
                'mem_rec': mem_class_out_rec,
                'spk_rec': spk_class_hidden_rec
            },
            'regression': {
                'mem_rec': mem_reg_out_rec,
                'spk_rec': spk_reg_hidden_rec
            },
            'shared': {
                'spk_rec': spk_shared_rec
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

SHARED LAYER (Purple - Common Processing):
  Input Size:              {self.input_size}
  Shared Hidden Layer:     {self.shared_hidden} LIF neurons

CLASSIFICATION BRANCH (Red - Analyte Classification):
  Classification Hidden:   {self.classification_hidden} LIF neurons
  Output:                  {self.num_classes} classes (Wine Quality)

REGRESSION BRANCH (Teal - Concentration Estimation):
  Regression Hidden:       {self.regression_hidden} LIF neurons
  Output:                  1 neuron (Ethanol Concentration)

PARAMETERS:
  Total:                   {total:,}
  Trainable:               {trainable:,}
{'='*80}
"""
        return summary
