# Weight Freezing Guide for Dual Encoder Multitask SNN

## Overview
The dual encoder multitask SNN now includes comprehensive weight freezing functionality to prevent task interference during training.

## Architecture Components
- **Classification Encoder**: 8-neuron encoder for classification task
- **Regression Encoder**: 8-neuron encoder for regression task  
- **Shared Layer**: 24-neuron layer that processes combined encoder outputs
- **Classification Branch**: Task-specific layers for wine quality classification
- **Regression Branch**: Task-specific layers for ethanol concentration regression

## Freezing Methods

### 1. Task-Specific Freezing
```python
# Freeze regression while training classification
model.freeze_regression_branch()

# Freeze classification while training regression
model.freeze_classification_branch()

# Freeze shared layer (use with caution)
model.freeze_shared_layer()
```

### 2. Unfreezing Methods
```python
# Unfreeze specific branches
model.unfreeze_regression_branch()
model.unfreeze_classification_branch()
model.unfreeze_shared_layer()

# Unfreeze everything for full multitask training
model.unfreeze_all()
```

### 3. Parameter Monitoring
```python
# Get detailed parameter counts
params = model.get_trainable_parameters()

# Print comprehensive status report
model.print_parameter_status()
```

## Training Strategies

### Sequential Training
```python
# 1. Train classification first
model.freeze_regression_branch()
optimizer_class = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.001
)
# ... train classification ...

# 2. Train regression second
model.unfreeze_regression_branch()
model.freeze_classification_branch()
optimizer_reg = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.001
)
# ... train regression ...

# 3. Fine-tune both together
model.unfreeze_all()
optimizer_multi = torch.optim.Adam(model.parameters(), lr=0.0001)
# ... train both tasks ...
```

### Task-Specific Fine-tuning
```python
# Load pre-trained model
model.load_state_dict(torch.load('pretrained_model.pth'))

# Fine-tune only classification on new data
model.freeze_regression_branch()
# ... train with classification data ...

# The regression performance remains unchanged
```

## Parameter Distribution
- **Total Parameters**: 1,388
- **Classification Encoder**: 56 params (6×8 + 8 bias)
- **Regression Encoder**: 56 params (6×8 + 8 bias)  
- **Shared Layer**: 408 params (16×24 + 24 bias)
- **Classification Branch**: 451 params (24×16 + 16×3 + biases)
- **Regression Branch**: 417 params (24×16 + 16×1 + biases)

## Benefits
1. **Task Isolation**: Train one task without affecting the other
2. **Catastrophic Forgetting Prevention**: Maintain learned representations
3. **Flexible Training**: Different learning rates per task
4. **Sequential Learning**: Add new tasks without losing old ones
5. **Debugging**: Isolate task-specific issues

## Usage Examples
See `demo_weight_freezing.py` for complete training examples with all three scenarios.