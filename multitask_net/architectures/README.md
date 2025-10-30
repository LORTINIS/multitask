# Multitask SNN Architectures

This folder contains different architecture implementations for multitask SNNs.

## Available Architectures

### `shared_28_14_class_8_reg_1.py`
- **Description**: Two shared layers (28→14) with separate classification (8→3) and regression (→1) branches
- **Architecture Name**: `shared_28_14_class_8_reg_1`
- **Training Script**: `train_two_shared_layers.py`
- **Shared Layers**: 
  - Hidden 1: 28 LIF neurons
  - Hidden 2: 14 LIF neurons
- **Classification Branch**: 
  - Hidden: 8 LIF neurons
  - Output: 3 neurons (HQ, LQ, AQ wine quality classes)
- **Regression Branch**: 
  - Output: 1 neuron (ethanol concentration, direct from shared layer 2)

### `shared_28_class_8_reg_14.py`
- **Description**: Single shared layer (28) with separate hidden layers for classification (8) and regression (14)
- **Architecture Name**: `shared_28_class_8_reg_14`
- **Training Script**: `train_single_shared_layer.py`
- **Shared Layer**: 
  - Hidden: 28 LIF neurons
- **Classification Branch**: 
  - Hidden: 8 LIF neurons
  - Output: 3 neurons (HQ, LQ, AQ wine quality classes)
- **Regression Branch**: 
  - Hidden: 14 LIF neurons
  - Output: 1 neuron (ethanol concentration)

## Training Script Selection

Each architecture specifies which training script to use via the `TRAINING_SCRIPT` class variable. The dispatcher (`multitask_classification_regression.py`) automatically routes to the correct training script based on the architecture selected.

## Adding New Architectures

To add a new architecture:

1. Create a new Python file in this directory (e.g., `my_architecture.py`)

2. Implement a class named `MultitaskSNN` that inherits from `nn.Module`

3. Add `ARCHITECTURE_NAME` and `TRAINING_SCRIPT` class variables:
   ```python
   class MultitaskSNN(nn.Module):
       ARCHITECTURE_NAME = "my_unique_architecture_name"
       TRAINING_SCRIPT = "train_my_architecture.py"  # or use existing script
       
       def __init__(self, input_size, ...):
           # Your architecture implementation
   ```

4. Ensure your `forward()` method returns a dictionary with the following structure:
   ```python
   return {
       'classification': {
           'mem_rec': mem_class_out_rec,
           'spk_rec': spk_class_rec
       },
       'regression': {
           'mem_rec': mem_reg_out_rec,
           'spk_rec': spk_reg_rec  # Optional: task-specific hidden layer spikes
       },
       'shared': {
           'spk_rec': spk_shared_rec  # For single shared layer
           # OR
           'spk_rec1': spk_shared1_rec,  # For multiple shared layers
           'spk_rec2': spk_shared2_rec
       }
   }
   ```

5. Create a corresponding training script (or use an existing one that matches your architecture's parameter requirements)

6. The architecture will be automatically discovered by the dispatcher

## Architecture Naming Convention

Use descriptive names that reflect the architecture structure:
- Format: `shared_H1_H2_class_C_reg_R`
- Example: `shared_28_14_class_8_reg_1`
  - `shared_28_14`: Shared layers with 28 and 14 neurons
  - `class_8`: Classification branch with 8 hidden neurons
  - `reg_1`: Regression branch with 1 output

## Usage

The main dispatcher script will automatically discover all architectures and route to the correct training script:

```bash
python multitask_classification_regression.py
```

This will:
1. Display all available architectures
2. Let you select one
3. Automatically load the appropriate training script
4. Train the model with the correct parameters

Alternatively, you can run training scripts directly:

```bash
# For two shared layer architectures
python train_two_shared_layers.py

# For single shared layer architectures  
python train_single_shared_layer.py
```

The selected architecture name will be included in the saved model filename.
