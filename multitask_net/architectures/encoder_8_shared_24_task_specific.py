"""
Multitask Spiking Neural Network Architecture: Dual-Encoder-8 + Shared-24 + Task-Specific

Architecture Overview:
- Two separate 8-neuron encoders (one for classification, one for regression) 
- Combined encoder outputs (16 neurons) feed into shared 24-neuron layer
- Task-specific branches: classification (24→16→3) and regression (24→16→1)
- Design philosophy: Task-specific feature extraction with shared representation learning
"""

import torch
import torch.nn as nn
import snntorch as snn


class MultitaskSNN(nn.Module):
    """
    Multitask SNN with dual encoders and shared processing layer.
    
    Architecture: Two separate 8-neuron encoders → 16→24 shared layer → task-specific branches
    """
    
    ARCHITECTURE_NAME = "dual_encoder_8_shared_24_task_specific"
    TRAINING_SCRIPT = "train_single_shared_layer_time_series.py"

    def __init__(
        self,
        input_size=6,
        encoder_hidden=8,
        shared_hidden=24,
        classification_hidden=16,
        regression_hidden=16,
        num_classes=3,
        beta=0.9,
        spike_grad=None,
        dropout_rate=0.0
    ):
        """
        Initialize dual encoder multitask SNN.
        
        Args:
            input_size: Number of input features (6 MQ sensors)
            encoder_hidden: Size of each encoder (8 neurons each)
            shared_hidden: Size of shared layer (24 neurons)
            classification_hidden: Hidden layer size for classification branch
            regression_hidden: Hidden layer size for regression branch
            num_classes: Number of output classes for classification
            beta: Leak parameter for LIF neurons
            spike_grad: Spike gradient function (default: snntorch default)
            dropout_rate: Not used in SNN context
        """
        super(MultitaskSNN, self).__init__()

        # Store architecture parameters
        self.input_size = input_size
        self.encoder_hidden = encoder_hidden
        self.shared_hidden = shared_hidden
        self.classification_hidden = classification_hidden
        self.regression_hidden = regression_hidden
        self.num_classes = num_classes

        # ================================================================
        # DUAL ENCODER LAYERS (Input → 8 neurons each)
        # ================================================================
        # Classification encoder
        self.fc_encoder_class = nn.Linear(input_size, encoder_hidden)
        self.lif_encoder_class = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Regression encoder  
        self.fc_encoder_reg = nn.Linear(input_size, encoder_hidden)
        self.lif_encoder_reg = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ================================================================
        # SHARED LAYER (16 → 24 neurons)
        # ================================================================
        self.fc_shared = nn.Linear(encoder_hidden * 2, shared_hidden)  # 8+8=16 → 24
        self.lif_shared = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ================================================================
        # CLASSIFICATION BRANCH (24 → 16 → 3)
        # ================================================================
        self.fc_class_hidden = nn.Linear(shared_hidden, classification_hidden)
        self.lif_class_hidden = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_class_out = nn.Linear(classification_hidden, num_classes)
        self.lif_class_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ================================================================
        # REGRESSION BRANCH (24 → 16 → 1)
        # ================================================================
        self.fc_reg_hidden = nn.Linear(shared_hidden, regression_hidden)
        self.lif_reg_hidden = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc_reg_out = nn.Linear(regression_hidden, 1)
        self.lif_reg_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """
        Forward pass through dual encoder multitask SNN.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, input_size)
            
        Returns:
            Dictionary containing outputs from all network components:
            - classification: {'mem_rec': membrane potentials, 'spk_rec': spike recordings}
            - regression: {'mem_rec': membrane potentials, 'spk_rec': spike recordings}  
            - shared: {'spk_rec': shared layer spike recordings}
            - encoder_class: {'spk_rec': classification encoder spikes}
            - encoder_reg: {'spk_rec': regression encoder spikes}
        """
        # Initialize membrane potentials for all layers
        mem_encoder_class = self.lif_encoder_class.init_leaky()
        mem_encoder_reg = self.lif_encoder_reg.init_leaky()
        mem_shared = self.lif_shared.init_leaky()
        mem_class_hidden = self.lif_class_hidden.init_leaky()
        mem_class_out = self.lif_class_out.init_leaky()
        mem_reg_hidden = self.lif_reg_hidden.init_leaky()
        mem_reg_out = self.lif_reg_out.init_leaky()
        
        # Recording lists
        spk_encoder_class_rec = []
        spk_encoder_reg_rec = []
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
            # DUAL ENCODER LAYERS (Input → 8 neurons each)
            # ================================================================
            # Classification encoder
            cur_encoder_class = self.fc_encoder_class(x_step)
            spk_encoder_class, mem_encoder_class = self.lif_encoder_class(cur_encoder_class, mem_encoder_class)
            
            # Regression encoder
            cur_encoder_reg = self.fc_encoder_reg(x_step)
            spk_encoder_reg, mem_encoder_reg = self.lif_encoder_reg(cur_encoder_reg, mem_encoder_reg)
            
            # ================================================================
            # SHARED LAYER (16 → 24 neurons) - concatenate encoder outputs
            # ================================================================
            combined_encoders = torch.cat([spk_encoder_class, spk_encoder_reg], dim=1)  # [batch, 16]
            cur_shared = self.fc_shared(combined_encoders)
            spk_shared, mem_shared = self.lif_shared(cur_shared, mem_shared)
            
            # ================================================================
            # CLASSIFICATION TASK BRANCH (24 → 16 → 3)
            # ================================================================
            cur_class_hidden = self.fc_class_hidden(spk_shared)
            spk_class_hidden, mem_class_hidden = self.lif_class_hidden(cur_class_hidden, mem_class_hidden)
            
            cur_class_out = self.fc_class_out(spk_class_hidden)
            spk_class_out, mem_class_out = self.lif_class_out(cur_class_out, mem_class_out)
            
            # ================================================================
            # REGRESSION TASK BRANCH (24 → 16 → 1)
            # ================================================================
            cur_reg_hidden = self.fc_reg_hidden(spk_shared)
            spk_reg_hidden, mem_reg_hidden = self.lif_reg_hidden(cur_reg_hidden, mem_reg_hidden)
            
            cur_reg_out = self.fc_reg_out(spk_reg_hidden)
            spk_reg_out, mem_reg_out = self.lif_reg_out(cur_reg_out, mem_reg_out)
            
            # Record activity
            spk_encoder_class_rec.append(spk_encoder_class)
            spk_encoder_reg_rec.append(spk_encoder_reg)
            spk_shared_rec.append(spk_shared)
            spk_class_hidden_rec.append(spk_class_hidden)
            mem_class_out_rec.append(mem_class_out)
            spk_reg_hidden_rec.append(spk_reg_hidden)
            mem_reg_out_rec.append(mem_reg_out)
        
        # Stack recordings
        spk_encoder_class_rec = torch.stack(spk_encoder_class_rec)
        spk_encoder_reg_rec = torch.stack(spk_encoder_reg_rec)
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
            },
            'encoder_class': {
                'spk_rec': spk_encoder_class_rec
            },
            'encoder_reg': {
                'spk_rec': spk_encoder_reg_rec
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

DUAL ENCODERS (Task-Specific Input Processing):
  Input Size:              {self.input_size} (MQ sensors only)
  Classification Encoder:  {self.encoder_hidden} LIF neurons
  Regression Encoder:      {self.encoder_hidden} LIF neurons

SHARED LAYER (Common Representation):
  Shared Hidden Layer:     {self.shared_hidden} LIF neurons ({self.encoder_hidden}+{self.encoder_hidden} → {self.shared_hidden})

CLASSIFICATION BRANCH (Wine Quality):
  Classification Hidden:   {self.classification_hidden} LIF neurons
  Output:                  {self.num_classes} classes (HQ, LQ, AQ)

REGRESSION BRANCH (Ethanol Concentration):
  Regression Hidden:       {self.regression_hidden} LIF neurons
  Output:                  1 neuron (Continuous Concentration)

LAYER CONNECTIONS:
  {self.input_size} → [{self.encoder_hidden}, {self.encoder_hidden}] → {self.shared_hidden} → {{
    Classification: {self.shared_hidden} → {self.classification_hidden} → {self.num_classes}
    Regression:     {self.shared_hidden} → {self.regression_hidden} → 1
  }}

PARAMETERS:
  Total:                   {total:,}
  Trainable:               {trainable:,}
{'='*80}
"""
        return summary
    
    def get_layer_sizes(self):
        """Return dictionary of layer sizes for compatibility."""
        return {
            'input_size': self.input_size,
            'encoder_hidden': self.encoder_hidden,
            'shared_hidden': self.shared_hidden,
            'classification_hidden': self.classification_hidden,
            'regression_hidden': self.regression_hidden,
            'num_classes': self.num_classes
        }
    
    def freeze_classification_branch(self):
        """
        Freeze classification branch parameters to prevent updates during regression training.
        This includes the classification encoder and classification-specific layers.
        """
        # Freeze classification encoder
        for param in self.fc_encoder_class.parameters():
            param.requires_grad = False
        
        # Freeze classification branch layers
        for param in self.fc_class_hidden.parameters():
            param.requires_grad = False
        for param in self.fc_class_out.parameters():
            param.requires_grad = False
        
        print("🔒 Classification branch frozen (encoder + classification layers)")
    
    def freeze_regression_branch(self):
        """
        Freeze regression branch parameters to prevent updates during classification training.
        This includes the regression encoder and regression-specific layers.
        """
        # Freeze regression encoder
        for param in self.fc_encoder_reg.parameters():
            param.requires_grad = False
        
        # Freeze regression branch layers
        for param in self.fc_reg_hidden.parameters():
            param.requires_grad = False
        for param in self.fc_reg_out.parameters():
            param.requires_grad = False
        
        print("🔒 Regression branch frozen (encoder + regression layers)")
    
    def freeze_shared_layer(self):
        """
        Freeze shared layer parameters to prevent updates.
        Use with caution as this affects both tasks.
        """
        for param in self.fc_shared.parameters():
            param.requires_grad = False
        print("🔒 Shared layer frozen")
    
    def unfreeze_classification_branch(self):
        """
        Unfreeze classification branch parameters to allow updates.
        """
        # Unfreeze classification encoder
        for param in self.fc_encoder_class.parameters():
            param.requires_grad = True
        
        # Unfreeze classification branch layers
        for param in self.fc_class_hidden.parameters():
            param.requires_grad = True
        for param in self.fc_class_out.parameters():
            param.requires_grad = True
        
        print("🔓 Classification branch unfrozen")
    
    def unfreeze_regression_branch(self):
        """
        Unfreeze regression branch parameters to allow updates.
        """
        # Unfreeze regression encoder
        for param in self.fc_encoder_reg.parameters():
            param.requires_grad = True
        
        # Unfreeze regression branch layers
        for param in self.fc_reg_hidden.parameters():
            param.requires_grad = True
        for param in self.fc_reg_out.parameters():
            param.requires_grad = True
        
        print("🔓 Regression branch unfrozen")
    
    def unfreeze_shared_layer(self):
        """
        Unfreeze shared layer parameters to allow updates.
        """
        for param in self.fc_shared.parameters():
            param.requires_grad = True
        print("🔓 Shared layer unfrozen")
    
    def unfreeze_all(self):
        """
        Unfreeze all parameters to allow full multitask training.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 All parameters unfrozen - full multitask training enabled")
    
    def get_trainable_parameters(self):
        """
        Get count of trainable parameters by component.
        
        Returns:
            Dictionary with parameter counts for each component
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'classification_encoder': count_params(self.fc_encoder_class),
            'regression_encoder': count_params(self.fc_encoder_reg),
            'shared_layer': count_params(self.fc_shared),
            'classification_hidden': count_params(self.fc_class_hidden),
            'classification_output': count_params(self.fc_class_out),
            'regression_hidden': count_params(self.fc_reg_hidden),
            'regression_output': count_params(self.fc_reg_out),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'total_frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad)
        }
    
    def print_parameter_status(self):
        """
        Print detailed status of trainable/frozen parameters.
        """
        params = self.get_trainable_parameters()
        total_params = params['total_trainable'] + params['total_frozen']
        
        print("\n🔍 Parameter Status Report:")
        print("=" * 50)
        print(f"Classification Encoder:  {params['classification_encoder']:>6,} params")
        print(f"Regression Encoder:      {params['regression_encoder']:>6,} params")
        print(f"Shared Layer:            {params['shared_layer']:>6,} params")
        print(f"Classification Branch:   {params['classification_hidden'] + params['classification_output']:>6,} params")
        print(f"Regression Branch:       {params['regression_hidden'] + params['regression_output']:>6,} params")
        print("-" * 50)
        print(f"Total Trainable:         {params['total_trainable']:>6,} params")
        print(f"Total Frozen:            {params['total_frozen']:>6,} params")
        print(f"Total Parameters:        {total_params:>6,} params")
        print(f"Trainable Ratio:         {params['total_trainable']/total_params*100:>5.1f}%")
        print("=" * 50)


def create_model(config):
    """
    Factory function to create MultitaskSNN from configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration parameters
        
    Returns:
        MultitaskSNN instance
    """
    return MultitaskSNN(
        input_size=config.get('input_size', 6),
        encoder_hidden=config.get('encoder_hidden', 8),
        shared_hidden=config.get('shared_hidden', 24), 
        classification_hidden=config.get('classification_hidden', 16),
        regression_hidden=config.get('regression_hidden', 16),
        num_classes=config.get('num_classes', 3),
        beta=config.get('beta', 0.9),
        spike_grad=config.get('spike_grad', None),
        dropout_rate=config.get('dropout_rate', 0.0)
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Dual Encoder Multitask SNN Architecture")
    
    # Create model with default parameters
    model = MultitaskSNN()
    
    # Print architecture summary
    print(model.get_architecture_summary())
    
    # Test with sample data
    batch_size = 2
    seq_length = 10
    input_size = 6
    
    x = torch.randn(seq_length, batch_size, input_size)
    
    print(f"\nTesting with input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print("Output shapes:")
    for key, value in output.items():
        if isinstance(value, dict):
            for subkey, tensor in value.items():
                print(f"  {key}.{subkey}: {tensor.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    print(f"\nModel has {model.get_num_parameters()[0]:,} total parameters")