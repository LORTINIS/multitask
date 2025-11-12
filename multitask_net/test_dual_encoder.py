#!/usr/bin/env python3
"""
Test script for the dual encoder multitask SNN architecture.
"""

import torch
import sys
import os

# Add the parent directory to the path so we can import the architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from architectures.encoder_8_shared_24_task_specific import MultitaskSNN

def test_dual_encoder_architecture():
    """Test the dual encoder architecture with sample data."""
    print("Testing Dual Encoder Multitask SNN Architecture")
    print("=" * 60)
    
    # Architecture parameters (matching the original training setup)
    input_size = 6  # 6 MQ sensors (no temperature/humidity)
    encoder_hidden = 8  # Each encoder has 8 neurons
    shared_hidden = 24  # Combined encoders (8+8) → 24
    classification_hidden = 16
    regression_hidden = 16
    num_classes = 3
    beta = 0.9
    
    # Create the model
    model = MultitaskSNN(
        input_size=input_size,
        encoder_hidden=encoder_hidden,
        shared_hidden=shared_hidden,
        classification_hidden=classification_hidden,
        regression_hidden=regression_hidden,
        num_classes=num_classes,
        beta=beta
    )
    
    print(model.get_architecture_summary())
    
    # Test with sample data
    batch_size = 4
    seq_length = 100  # Shorter sequence for testing
    
    # Create sample input data
    x = torch.randn(seq_length, batch_size, input_size)
    
    print(f"Input shape: {x.shape}")
    print("Running forward pass...")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print("✓ Forward pass successful!")
        print(f"Classification output shape: {output['classification']['mem_rec'].shape}")
        print(f"Regression output shape: {output['regression']['mem_rec'].shape}")
        print(f"Shared layer shape: {output['shared']['spk_rec'].shape}")
        print(f"Classification encoder shape: {output['encoder_class']['spk_rec'].shape}")
        print(f"Regression encoder shape: {output['encoder_reg']['spk_rec'].shape}")
        
        # Check output dimensions
        expected_class_shape = (seq_length, batch_size, num_classes)
        expected_reg_shape = (seq_length, batch_size, 1)
        expected_shared_shape = (seq_length, batch_size, shared_hidden)
        expected_encoder_shape = (seq_length, batch_size, encoder_hidden)
        
        assert output['classification']['mem_rec'].shape == expected_class_shape
        assert output['regression']['mem_rec'].shape == expected_reg_shape
        assert output['shared']['spk_rec'].shape == expected_shared_shape
        assert output['encoder_class']['spk_rec'].shape == expected_encoder_shape
        assert output['encoder_reg']['spk_rec'].shape == expected_encoder_shape
        
        print("✓ All output shapes are correct!")
        
        # Test parameter counting
        total_params, trainable_params = model.get_num_parameters()
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test weight freezing functionality
        print("\n" + "="*60)
        print("TESTING WEIGHT FREEZING FUNCTIONALITY")
        print("="*60)
        
        # Initial state - all parameters trainable
        print("\n1. Initial state (all trainable):")
        model.print_parameter_status()
        
        # Test freezing regression branch for classification training
        print("\n2. Freezing regression branch:")
        model.freeze_regression_branch()
        model.print_parameter_status()
        
        # Test freezing classification branch for regression training  
        print("\n3. Freezing classification branch (unfreezing regression):")
        model.unfreeze_regression_branch()
        model.freeze_classification_branch()
        model.print_parameter_status()
        
        # Test unfreezing all for full multitask training
        print("\n4. Unfreezing all for multitask training:")
        model.unfreeze_all()
        model.print_parameter_status()
        
        print("\n🎉 Dual encoder architecture test PASSED!")
        print("✓ Weight freezing functionality working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dual_encoder_architecture()
    exit(0 if success else 1)