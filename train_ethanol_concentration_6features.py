"""
Train Ethanol Concentration Regression SNN (6 features, no temp/humidity)
==========================================================================

This script trains an ethanol concentration regression SNN using:
- 6 MQ sensor features ONLY (no temperature/humidity)
- Sequence length 1000
- Best hyperparameters from HPO
- Direct spike encoding

Matches the data format expected by tsa_singletask_concentration_regression.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import snntorch as snn
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# CONFIGURATION (Best HPO Parameters)
# ============================================================================

# From best_params_20251110_045632.txt
BETA = 0.9320405786091646
LEARNING_RATE = 0.009478855050990368
BATCH_SIZE = 8
# DROPOUT = 0.0359117786731738  # Not used in basic model
NUM_EPOCHS = 100
SEQUENCE_LENGTH = 1000
DOWNSAMPLE_FACTOR = 2

# Model architecture (from existing model)
HIDDEN_SIZE1 = 28
HIDDEN_SIZE2 = 14
INPUT_SIZE = 6  # Only MQ sensors
OUTPUT_SIZE = 1  # Regression

print(f"Configuration:")
print(f"  Beta: {BETA}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE1} -> {HIDDEN_SIZE2} -> {OUTPUT_SIZE}")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class TimeSeriesConcentrationSNN(nn.Module):
    """
    SNN for ethanol concentration regression (6 features, no temp/humidity)
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, beta=0.9):
        super().__init__()
        
        # Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        # LIF neurons
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        """
        Forward pass through time series SNN.
        
        Args:
            x: Input tensor [timesteps, batch, features]
            
        Returns:
            - mem3_rec: Output membrane potentials [timesteps, batch, 1]
            - spk3_rec: Output spikes [timesteps, batch, 1] 
            - spk1_rec: Hidden layer 1 spikes [timesteps, batch, hidden1]
            - spk2_rec: Hidden layer 2 spikes [timesteps, batch, hidden2]
        """
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record spikes and membrane potentials
        spk1_list, spk2_list, spk3_list, mem3_list = [], [], [], []
        
        # Process each timestep
        for t in range(x.size(0)):
            # Layer 1
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output layer (regression)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Record
            spk1_list.append(spk1)
            spk2_list.append(spk2)
            spk3_list.append(spk3)
            mem3_list.append(mem3)
            
        return (
            torch.stack(mem3_list),  # [T, B, 1]
            torch.stack(spk3_list),  # [T, B, 1] 
            torch.stack(spk1_list),  # [T, B, H1]
            torch.stack(spk2_list)   # [T, B, H2]
        )

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("LOADING ETHANOL SPIKE DATA (6 features)")
print("="*80)

# Load spike data
spike_train = np.load('data/spike_data_ethanol/spike_train.npy')
spike_test = np.load('data/spike_data_ethanol/spike_test.npy')
y_train = np.load('data/spike_data_ethanol/y_train.npy')
y_test = np.load('data/spike_data_ethanol/y_test.npy')
config = np.load('data/spike_data_ethanol/config.npy', allow_pickle=True).item()

print(f"Spike train shape: {spike_train.shape}")
print(f"Spike test shape: {spike_test.shape}")
print(f"Train labels: {y_train.shape}, range: [{y_train.min():.1f}%, {y_train.max():.1f}%]")
print(f"Test labels: {y_test.shape}, range: [{y_test.min():.1f}%, {y_test.max():.1f}%]")
print(f"Features: {config['feature_names']}")

# Normalize targets for training
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"Normalized targets: [{y_train_norm.min():.3f}, {y_train_norm.max():.3f}]")

# Convert to tensors
spike_train_tensor = torch.FloatTensor(spike_train).permute(1, 0, 2)  # [T, B, F]
spike_test_tensor = torch.FloatTensor(spike_test).permute(1, 0, 2)    # [T, B, F]
y_train_tensor = torch.FloatTensor(y_train_norm)
y_test_tensor = torch.FloatTensor(y_test_norm)

print(f"Tensor shapes:")
print(f"  spike_train: {spike_train_tensor.shape} [timesteps, batch, features]")
print(f"  spike_test: {spike_test_tensor.shape} [timesteps, batch, features]")

# ============================================================================
# MODEL INSTANTIATION
# ============================================================================

print("\n" + "="*80)
print("MODEL INSTANTIATION")
print("="*80)

model = TimeSeriesConcentrationSNN(
    input_size=INPUT_SIZE,
    hidden_size1=HIDDEN_SIZE1,
    hidden_size2=HIDDEN_SIZE2,
    output_size=OUTPUT_SIZE,
    beta=BETA
).to(device)

print(f"Model: {model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("TRAINING SETUP")
print("="*80)

# Loss and optimizer (use AdamW as per HPO results)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Data loaders
train_dataset = TensorDataset(spike_train_tensor.permute(1, 0, 2), y_train_tensor)
test_dataset = TensorDataset(spike_test_tensor.permute(1, 0, 2), y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("TRAINING")
print("="*80)

train_losses = []
test_losses = []
test_r2_scores = []

best_test_loss = float('inf')
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.permute(1, 0, 2).to(device)  # [T, B, F]
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        mem_rec, _, _, _ = model(data)
        
        # Average membrane potentials over time (same as original training)
        predictions = mem_rec.mean(dim=0).squeeze()  # [B]
        
        # Handle single sample case
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.permute(1, 0, 2).to(device)  # [T, B, F]
            targets = targets.to(device)
            
            # Forward pass
            mem_rec, _, _, _ = model(data)
            
            # Average membrane potentials over time
            predictions = mem_rec.mean(dim=0).squeeze()  # [B]
            
            # Handle single sample case
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            
            # Compute loss
            loss = criterion(predictions, targets)
            test_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    # Calculate R² score
    r2 = r2_score(all_targets, all_predictions)
    test_r2_scores.append(r2)
    
    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f} | "
              f"R²: {r2:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

print(f"\nTraining completed!")
print(f"Best test loss: {best_test_loss:.6f}")

# Load best model
model.load_state_dict(best_model_state)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

model.eval()
final_predictions = []
final_targets = []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.permute(1, 0, 2).to(device)
        targets = targets.to(device)
        
        # Forward pass
        mem_rec, _, _, _ = model(data)
        predictions = mem_rec.mean(dim=0).squeeze()
        
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0)
            
        final_predictions.extend(predictions.cpu().numpy())
        final_targets.extend(targets.cpu().numpy())

final_predictions = np.array(final_predictions)
final_targets = np.array(final_targets)

# Denormalize to original scale
final_predictions_original = scaler_y.inverse_transform(
    final_predictions.reshape(-1, 1)).flatten()
final_targets_original = scaler_y.inverse_transform(
    final_targets.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(final_targets_original, final_predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(final_targets_original, final_predictions_original)
r2 = r2_score(final_targets_original, final_predictions_original)

print(f"\nFinal Performance:")
print(f"  RMSE: {rmse:.4f}%")
print(f"  MAE: {mae:.4f}%")
print(f"  R² Score: {r2:.4f}")
print(f"  Error rate: {(mae/final_targets_original.mean())*100:.2f}% of mean")

print(f"\nPrediction vs True:")
for i in range(len(final_predictions_original)):
    print(f"  Sample {i}: pred={final_predictions_original[i]:.2f}%, true={final_targets_original[i]:.2f}%, error={abs(final_predictions_original[i]-final_targets_original[i]):.2f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_dir = "trained_models"
os.makedirs(model_dir, exist_ok=True)

model_filename = (
    f"ethanol_concentration_snn_"
    f"6features_seq{SEQUENCE_LENGTH}_"
    f"beta{BETA:.4f}_"
    f"bs{BATCH_SIZE}_"
    f"lr{LEARNING_RATE:.6f}_"
    f"ep{NUM_EPOCHS}.pth"
)

model_path = os.path.join(model_dir, model_filename)

# Save with all necessary information for TSA script
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': INPUT_SIZE,
    'hidden1': HIDDEN_SIZE1,
    'hidden2': HIDDEN_SIZE2,
    'output_size': OUTPUT_SIZE,
    'beta': BETA,
    'scaler_y': scaler_y,
    'sequence_length': SEQUENCE_LENGTH,
    'downsample_factor': DOWNSAMPLE_FACTOR,
    'final_metrics': {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'test_loss': best_test_loss
    },
    'training_config': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'optimizer': 'AdamW',
        'features': 'MQ_sensors_only',
        'includes_temperature_humidity': False
    }
}, model_path)

print(f"✓ Model saved to: {model_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("TRAINING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training curves
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Progress')
axes[0].legend()
axes[0].grid(alpha=0.3)

# R² scores
axes[1].plot(test_r2_scores, label='Test R²', linewidth=2, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('R² Score')
axes[1].set_title('R² Score Over Time')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Predictions vs targets
axes[2].scatter(final_targets_original, final_predictions_original, alpha=0.7, s=100)
axes[2].plot([final_targets_original.min(), final_targets_original.max()],
             [final_targets_original.min(), final_targets_original.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[2].set_xlabel('True Concentration (%)')
axes[2].set_ylabel('Predicted Concentration (%)')
axes[2].set_title('Predictions vs True Values')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✅ ETHANOL CONCENTRATION SNN TRAINING COMPLETE!")
print("="*80)
print(f"\nModel Summary:")
print(f"  Architecture: {INPUT_SIZE} features (MQ sensors only)")
print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Performance: R² = {r2:.4f}, MAE = {mae:.4f}%")
print(f"  Model file: {model_filename}")
print(f"\nReady for TSA analysis with:")
print(f"  python explainability/tsa/tsa_singletask_concentration_regression.py \\")
print(f"    --checkpoint ../../{model_path} \\")
print(f"    --spike_data_dir ../../data/spike_data_ethanol")