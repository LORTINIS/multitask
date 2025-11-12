#!/usr/bin/env python3
"""
Example training script demonstrating weight freezing for the dual encoder multitask SNN.

This script shows how to:
1. Train only the classification task while keeping regression frozen
2. Train only the regression task while keeping classification frozen  
3. Train both tasks together (full multitask learning)
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path so we can import the architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from architectures.encoder_8_shared_24_task_specific import MultitaskSNN

def demonstrate_freezing_training():
    """Demonstrate training with different freezing configurations."""
    
    print("🔬 DUAL ENCODER WEIGHT FREEZING TRAINING DEMO")
    print("=" * 60)
    
    # Create model
    model = MultitaskSNN(
        input_size=6,
        encoder_hidden=8,
        shared_hidden=24,
        classification_hidden=16,
        regression_hidden=16,
        num_classes=3,
        beta=0.9
    )
    
    # Create sample data
    batch_size = 8
    seq_length = 50
    x = torch.randn(seq_length, batch_size, 6)
    
    # Sample targets
    class_targets = torch.randint(0, 3, (batch_size,))  # 3 classes
    reg_targets = torch.randn(batch_size, 1)  # Continuous values
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    
    print(f"Sample data shape: {x.shape}")
    print(f"Classification targets shape: {class_targets.shape}")
    print(f"Regression targets shape: {reg_targets.shape}")
    
    # ================================================================
    # SCENARIO 1: Train only classification (freeze regression)
    # ================================================================
    print("\n" + "="*60)
    print("SCENARIO 1: CLASSIFICATION-ONLY TRAINING")
    print("="*60)
    
    model.freeze_regression_branch()
    model.print_parameter_status()
    
    # Create optimizer for unfrozen parameters only
    optimizer_class = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    
    print("\n🎯 Training classification task (regression frozen)...")
    for epoch in range(3):
        optimizer_class.zero_grad()
        
        # Forward pass
        output = model(x)
        
        # Classification loss only
        class_logits = output['classification']['mem_rec'][-1]  # Last time step
        class_loss = class_criterion(class_logits, class_targets)
        
        # Backward pass
        class_loss.backward()
        optimizer_class.step()
        
        print(f"Epoch {epoch+1}: Classification Loss = {class_loss.item():.4f}")
    
    # ================================================================
    # SCENARIO 2: Train only regression (freeze classification)
    # ================================================================
    print("\n" + "="*60)
    print("SCENARIO 2: REGRESSION-ONLY TRAINING")
    print("="*60)
    
    model.unfreeze_regression_branch()
    model.freeze_classification_branch()
    model.print_parameter_status()
    
    # Create optimizer for unfrozen parameters only
    optimizer_reg = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    
    print("\n🎯 Training regression task (classification frozen)...")
    for epoch in range(3):
        optimizer_reg.zero_grad()
        
        # Forward pass
        output = model(x)
        
        # Regression loss only
        reg_logits = output['regression']['mem_rec'][-1]  # Last time step
        reg_loss = reg_criterion(reg_logits, reg_targets)
        
        # Backward pass
        reg_loss.backward()
        optimizer_reg.step()
        
        print(f"Epoch {epoch+1}: Regression Loss = {reg_loss.item():.4f}")
    
    # ================================================================
    # SCENARIO 3: Train both tasks together (full multitask)
    # ================================================================
    print("\n" + "="*60)
    print("SCENARIO 3: FULL MULTITASK TRAINING")
    print("="*60)
    
    model.unfreeze_all()
    model.print_parameter_status()
    
    # Create optimizer for all parameters
    optimizer_multi = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🎯 Training both tasks together...")
    for epoch in range(3):
        optimizer_multi.zero_grad()
        
        # Forward pass
        output = model(x)
        
        # Both losses
        class_logits = output['classification']['mem_rec'][-1]
        reg_logits = output['regression']['mem_rec'][-1]
        
        class_loss = class_criterion(class_logits, class_targets)
        reg_loss = reg_criterion(reg_logits, reg_targets)
        
        # Combined loss (you can weight these differently)
        total_loss = class_loss + reg_loss
        
        # Backward pass
        total_loss.backward()
        optimizer_multi.step()
        
        print(f"Epoch {epoch+1}: Class Loss = {class_loss.item():.4f}, "
              f"Reg Loss = {reg_loss.item():.4f}, Total = {total_loss.item():.4f}")
    
    print("\n✅ Training demonstration completed!")
    print("\n💡 Key Benefits of Weight Freezing:")
    print("   • Prevents task interference during single-task fine-tuning")
    print("   • Allows sequential training of tasks")
    print("   • Enables task-specific learning rates and strategies")
    print("   • Maintains learned representations when adding new tasks")

if __name__ == "__main__":
    demonstrate_freezing_training()