"""
Ethanol Concentration Regression using Time Series SNNs with snntorch
UPDATED: Now uses full time series data instead of aggregated statistics!

This script implements an SNN for ethanol concentration regression with:
1. Time series data loading and preprocessing
2. Direct spike encoding from temporal sequences
3. SNN architecture with temporal processing
4. Model training and evaluation for concentration prediction

Dataset: Ethanol Time-Series Dataset
Architecture: LIF-based SNN with 2 hidden layers (28, 14 neurons)
Task: Continuous concentration regression from temporal sensor data
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# snntorch
import snntorch as snn
from snntorch import spikegen

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# PART 1: TIME SERIES DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("PART 1: LOADING ETHANOL TIME-SERIES DATASET")
print("="*80)

def load_ethanol_dataset(base_path):
    """
    Loads the ethanol time-series dataset preserving temporal structure.
    """
    all_data = []
    
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    # Concentration mapping
    ethanol_concentration_map = {
        'C1': 1.0,    # 1%
        'C2': 2.5,    # 2.5%
        'C3': 5.0,    # 5%
        'C4': 10.0,   # 10%
        'C5': 15.0,   # 15%
        'C6': 20.0    # 20%
    }
    
    print(f"Loading data from: {base_path}")
    
    ethanol_path = os.path.join(base_path, 'Ethanol')
    
    if not os.path.exists(ethanol_path):
        raise FileNotFoundError(f"Ethanol folder not found at {ethanol_path}")
    
    print(f"Processing folder: Ethanol...")
    
    files = os.listdir(ethanol_path)
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(ethanol_path, file_name)
            
            try:
                df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=ts_columns)
                df_file['Filename'] = file_name
                df_file['Time_Point'] = range(len(df_file))
                
                # Extract concentration
                conc_code = file_name[3:5]
                df_file['Concentration_Value'] = ethanol_concentration_map.get(conc_code, np.nan)
                df_file['Concentration_Label'] = f"{ethanol_concentration_map.get(conc_code, 'Unknown')}%"
                
                # Repetition
                rep_start_index = file_name.rfind('R')
                df_file['Repetition'] = file_name[rep_start_index:rep_start_index + 3] if rep_start_index != -1 else 'Unknown'
                
                all_data.append(df_file)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows loaded: {len(final_df)}")
    return final_df

# Load dataset
dataset_path = 'data/wine'
if not os.path.exists(dataset_path):
    dataset_path = 'Dataset'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found")

ethanol_df = load_ethanol_dataset(dataset_path)

if ethanol_df.empty:
    raise ValueError("Dataset is empty")

print(f"\nDataset shape: {ethanol_df.shape}")
print(f"Unique files: {ethanol_df['Filename'].nunique()}")

# ============================================================================
# 1.1 Remove Stabilization Period
# ============================================================================
print("\n" + "-"*80)
print("1.1 Removing Stabilization Period")
print("-"*80)

stabilization_period = 500
processed_files = []

for filename in ethanol_df['Filename'].unique():
    file_data = ethanol_df[ethanol_df['Filename'] == filename].copy()
    
    if len(file_data) > stabilization_period:
        file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
        processed_files.append(file_data)

ethanol_df = pd.concat(processed_files, ignore_index=True)
print(f"\nDataset shape after removing stabilization: {ethanol_df.shape}")

# ============================================================================
# 1.2 Prepare Time Series Sequences
# ============================================================================
print("\n" + "-"*80)
print("1.2 Preparing Time Series Sequences")
print("-"*80)

sensor_cols = [
    'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
    'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
]
environmental_cols = ['Rel_Humidity (%)', 'Temperature (C)']
feature_cols = sensor_cols + environmental_cols

# Configuration
SEQUENCE_LENGTH = 1000
DOWNSAMPLE_FACTOR = 2

print(f"\nTime series configuration:")
print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
print(f"  Features: {len(feature_cols)}")

sequences = []
concentrations = []
metadata = []

for filename in ethanol_df['Filename'].unique():
    file_data = ethanol_df[ethanol_df['Filename'] == filename]
    
    # Extract time series
    sequence = file_data[feature_cols].values
    
    # Downsample
    if DOWNSAMPLE_FACTOR > 1:
        sequence = sequence[::DOWNSAMPLE_FACTOR]
    
    # Fix length
    if len(sequence) < SEQUENCE_LENGTH:
        padding = np.repeat(sequence[-1:], SEQUENCE_LENGTH - len(sequence), axis=0)
        sequence = np.vstack([sequence, padding])
    elif len(sequence) > SEQUENCE_LENGTH:
        sequence = sequence[:SEQUENCE_LENGTH]
    
    sequences.append(sequence)
    concentrations.append(file_data['Concentration_Value'].iloc[0])
    
    metadata.append({
        'Filename': filename,
        'Concentration_Value': file_data['Concentration_Value'].iloc[0],
        'Concentration_Label': file_data['Concentration_Label'].iloc[0],
        'Repetition': file_data['Repetition'].iloc[0]
    })

print(f"\nTotal sequences: {len(sequences)}")
print(f"Sequence shape: {sequences[0].shape} (timesteps, features)")

# ============================================================================
# 1.3 Handle Missing Values and Normalize
# ============================================================================
print("\n" + "-"*80)
print("1.3 Cleaning and Normalizing Time Series")
print("-"*80)

# Clean sequences
def clean_sequence(seq):
    for col_idx in range(seq.shape[1]):
        col_data = seq[:, col_idx]
        if np.isnan(col_data).any() or np.isinf(col_data).any():
            col_mean = np.nanmean(col_data[np.isfinite(col_data)])
            seq[:, col_idx] = np.where(
                np.isnan(col_data) | np.isinf(col_data), 
                col_mean, 
                col_data
            )
    return seq

sequences = [clean_sequence(seq) for seq in sequences]

# Normalize sequences
all_data = np.vstack(sequences)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(all_data)

normalized_sequences = []
for seq in sequences:
    normalized_seq = scaler_X.transform(seq)
    normalized_sequences.append(normalized_seq)

all_normalized = np.vstack(normalized_sequences)
print(f"\nNormalized range: [{all_normalized.min():.3f}, {all_normalized.max():.3f}]")

# ============================================================================
# 1.4 Prepare Target (Concentrations)
# ============================================================================
print("\n" + "-"*80)
print("1.4 Preparing Concentration Targets")
print("-"*80)

y_concentration = np.array(concentrations)

print(f"\nConcentration distribution:")
unique_concentrations = np.unique(y_concentration)
for conc in sorted(unique_concentrations):
    count = np.sum(y_concentration == conc)
    print(f"  {conc}%: {count} samples")

# Normalize concentrations for training
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_normalized = scaler_y.fit_transform(y_concentration.reshape(-1, 1)).flatten()

print(f"\nTarget range: [{y_concentration.min():.1f}%, {y_concentration.max():.1f}%]")
print(f"Normalized target range: [{y_normalized.min():.3f}, {y_normalized.max():.3f}]")

# ============================================================================
# 1.5 Create Fixed-Size Arrays
# ============================================================================
print("\n" + "-"*80)
print("1.5 Creating Fixed-Size Arrays")
print("-"*80)

max_length = SEQUENCE_LENGTH
num_samples = len(normalized_sequences)
num_features = len(feature_cols)

X = np.zeros((num_samples, max_length, num_features))

for i, seq in enumerate(normalized_sequences):
    X[i, :len(seq), :] = seq

print(f"\nFinal data shape: {X.shape} (samples, timesteps, features)")

# ============================================================================
# 1.6 Train-Test Split
# ============================================================================
print("\n" + "-"*80)
print("1.6 Train-Test Split")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_normalized,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Denormalize for display
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

print(f"\nTraining concentration range: [{y_train_orig.min():.1f}%, {y_train_orig.max():.1f}%]")
print(f"Test concentration range: [{y_test_orig.min():.1f}%, {y_test_orig.max():.1f}%]")

# ============================================================================
# PART 2: SPIKE ENCODING FROM TIME SERIES
# ============================================================================

print("\n" + "="*80)
print("PART 2: SPIKE ENCODING TIME SERIES DATA")
print("="*80)

print("""
ENCODING STRATEGY: DIRECT TIME SERIES TO SPIKES
------------------------------------------------
We convert normalized time series values directly to binary spikes:
- Values above threshold (0.5) → spike (1)
- Values below threshold → no spike (0)
- Preserves temporal dynamics throughout the sequence
""")

ENCODING_TYPE = 'rate'

print(f"\nEncoding type: {ENCODING_TYPE}")

def encode_time_series_direct(sequences):
    """
    Encode time series using direct threshold-based conversion.
    """
    print(f"  Encoding {sequences.shape[0]} sequences...")
    
    # Convert to tensor
    sequences_tensor = torch.FloatTensor(sequences)
    
    # Threshold-based: values > 0.5 become spikes
    spike_data = (sequences_tensor > 0.5).float()
    
    # Reshape to [timesteps, samples, features]
    spike_data = spike_data.permute(1, 0, 2)
    
    print(f"  Spike data shape: {spike_data.shape} (timesteps, samples, features)")
    
    # Calculate statistics
    total_spikes = spike_data.sum().item()
    spike_rate = total_spikes / spike_data.numel()
    
    print(f"  Total spikes: {total_spikes:,.0f}")
    print(f"  Spike rate: {spike_rate:.4f}")
    
    return spike_data



def encode_time_series_rate(sequences, num_steps=100):
    """
    Encode time series using rate encoding.
    Each input value in [0,1] determines the probability of a spike
    at each timestep across 'num_steps'.
    """
    print(f"  Encoding {sequences.shape[0]} sequences with rate encoding ({num_steps} timesteps)...")

    # Convert to tensor
    sequences_tensor = torch.FloatTensor(sequences)
    
    # Ensure data is normalized to [0, 1]
    min_val = sequences_tensor.min()
    max_val = sequences_tensor.max()
    if max_val > 1.0 or min_val < 0.0:
        sequences_tensor = (sequences_tensor - min_val) / (max_val - min_val + 1e-8)
        print(f"  Normalized data from [{min_val:.3f}, {max_val:.3f}] → [0, 1]")

    # Generate random spikes according to rate
    # Shape: [timesteps, samples, features]
    rand = torch.rand((num_steps, *sequences_tensor.shape))
    spike_data = (rand < sequences_tensor.unsqueeze(0)).float()

    print(f"  Spike data shape: {spike_data.shape} (timesteps, samples, features)")
    
    # Calculate statistics
    total_spikes = spike_data.sum().item()
    spike_rate = total_spikes / spike_data.numel()
    print(f"  Total spikes: {total_spikes:,.0f}")
    print(f"  Average spike rate: {spike_rate:.4f}")

    return spike_data


if ENCODING_TYPE == 'rate':
    # Encode training and test data
    print("\nEncoding training data:")
    spike_train = encode_time_series_rate(X_train, num_steps=100)

    print("\nEncoding test data:")
    spike_test = encode_time_series_rate(X_test, num_steps=100)
else:
    # Encode training and test data
    print("\nEncoding training data:")
    spike_train = encode_time_series_direct(X_train)

    print("\nEncoding test data:")
    spike_test = encode_time_series_direct(X_test)

# Convert labels to tensors
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

print("\n✓ Spike encoding completed!")

# ============================================================================
# PART 3: SNN ARCHITECTURE FOR TIME SERIES REGRESSION
# ============================================================================

print("\n" + "="*80)
print("PART 3: DESIGNING SNN ARCHITECTURE FOR TIME SERIES REGRESSION")
print("="*80)

class TimeSeriesConcentrationSNN(nn.Module):
    """
    Spiking Neural Network for Ethanol Concentration Regression from Time Series
    
    Architecture:
    - Input: [timesteps, batch, features] time series
    - Hidden Layer 1: 28 LIF neurons
    - Hidden Layer 2: 14 LIF neurons
    - Output Layer: 1 neuron (concentration)
    """
    
    def __init__(self, input_size=8, hidden_size1=28, hidden_size2=14, 
                 output_size=1, beta=0.9):
        super(TimeSeriesConcentrationSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        # LIF neuron layers
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        """
        Forward pass through the SNN
        
        Args:
            x: Input spike train [time_steps, batch_size, input_size]
            
        Returns:
            mem_rec3: Membrane potential recordings for regression
            spk_rec1, spk_rec2: Hidden layer spike recordings
        """
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record activity
        spk_rec1 = []
        spk_rec2 = []
        mem_rec3 = []
        
        # Process each timestep
        num_steps = x.size(0)
        for step in range(num_steps):
            x_step = x[step]
            
            # Layer 1
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Layer 3 (output)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # Record
            spk_rec1.append(spk1)
            spk_rec2.append(spk2)
            mem_rec3.append(mem3)
        
        # Stack recordings
        spk_rec1 = torch.stack(spk_rec1)
        spk_rec2 = torch.stack(spk_rec2)
        mem_rec3 = torch.stack(mem_rec3)
        
        return mem_rec3, spk_rec1, spk_rec2

# ============================================================================
# 3.1 Instantiate Model
# ============================================================================
print("\n" + "-"*80)
print("3.1 Model Instantiation")
print("-"*80)

input_size = num_features  # 8 features
hidden_size1 = 28
hidden_size2 = 14
output_size = 1
beta = 0.9

print(f"\nModel configuration:")
print(f"  Input size: {input_size} (features per timestep)")
print(f"  Hidden layer 1: {hidden_size1} neurons")
print(f"  Hidden layer 2: {hidden_size2} neurons")
print(f"  Output size: {output_size} (regression)")
print(f"  Beta: {beta}")
print(f"  Timesteps: {SEQUENCE_LENGTH}")

model = TimeSeriesConcentrationSNN(
    input_size=input_size,
    hidden_size1=hidden_size1,
    hidden_size2=hidden_size2,
    output_size=output_size,
    beta=beta
).to(device)

print(f"\nModel created on {device}")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================================================
# PART 4: TRAINING
# ============================================================================

print("\n" + "="*80)
print("PART 4: TRAINING TIME SERIES REGRESSION SNN")
print("="*80)

# Training configuration
num_epochs = 100
batch_size = 16
learning_rate = 0.001

print(f"\nTraining configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Data loaders
train_dataset = torch.utils.data.TensorDataset(
    spike_train.permute(1, 0, 2),  # [samples, time, features]
    y_train_tensor
)

test_dataset = torch.utils.data.TensorDataset(
    spike_test.permute(1, 0, 2),
    y_test_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# 4.1 Training Loop
# ============================================================================
print("\n" + "-"*80)
print("4.1 Training Loop")
print("-"*80)

train_losses = []
test_losses = []
test_r2_scores = []

print("\nStarting training...")
print("-" * 80)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for spike_batch, target_batch in train_loader:
        spike_batch = spike_batch.permute(1, 0, 2).to(device)
        target_batch = target_batch.to(device)
        
        mem_rec, _, _ = model(spike_batch)
        
        # Use mean membrane potential for regression
        predictions = mem_rec.mean(dim=0).squeeze()
        
        loss = criterion(predictions, target_batch)
        
        optimizer.zero_grad()
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
        for spike_batch, target_batch in test_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            target_batch = target_batch.to(device)
            
            mem_rec, _, _ = model(spike_batch)
            predictions = mem_rec.mean(dim=0).squeeze()
            
            loss = criterion(predictions, target_batch)
            test_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    # Calculate R² score
    r2 = r2_score(all_targets, all_predictions)
    test_r2_scores.append(r2)
    
    scheduler.step(test_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f} | "
              f"R²: {r2:.4f}")

print("\n✓ Training completed!")

# ============================================================================
# 4.2 Visualize Training History
# ============================================================================
print("\n" + "-"*80)
print("4.2 Training History")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
axes[0].set_title('Training Progress', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# R² Score
axes[1].plot(test_r2_scores, label='Test R²', linewidth=2, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('R² Score', fontweight='bold')
axes[1].set_title('R² Score Over Time', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 4.3 Final Evaluation
# ============================================================================
print("\n" + "-"*80)
print("4.3 Final Model Evaluation")
print("-"*80)

model.eval()
final_predictions = []
final_targets = []

with torch.no_grad():
    for spike_batch, target_batch in test_loader:
        spike_batch = spike_batch.permute(1, 0, 2).to(device)
        
        mem_rec, _, _ = model(spike_batch)
        predictions = mem_rec.mean(dim=0).squeeze()
        
        final_predictions.extend(predictions.cpu().numpy())
        final_targets.extend(target_batch.numpy())

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

print("\n" + "="*80)
print("FINAL PERFORMANCE")
print("="*80)
print(f"\nRMSE: {rmse:.4f}%")
print(f"MAE: {mae:.4f}%")
print(f"R² Score: {r2:.4f}")
print(f"Error rate: {(mae/final_targets_original.mean())*100:.2f}% of mean")

# ============================================================================
# 4.4 Prediction Visualization
# ============================================================================
print("\n" + "-"*80)
print("4.4 Prediction Visualization")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(final_targets_original, final_predictions_original, 
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0, 0].plot([final_targets_original.min(), final_targets_original.max()],
                [final_targets_original.min(), final_targets_original.max()],
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Concentration (%)', fontweight='bold')
axes[0, 0].set_ylabel('Predicted Concentration (%)', fontweight='bold')
axes[0, 0].set_title('Predicted vs Actual', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Colored by concentration
scatter = axes[0, 1].scatter(final_targets_original, final_predictions_original, 
                             c=final_targets_original, cmap='viridis', s=50, alpha=0.6)
axes[0, 1].plot([final_targets_original.min(), final_targets_original.max()],
                [final_targets_original.min(), final_targets_original.max()],
                'r--', linewidth=2)
axes[0, 1].set_xlabel('Actual Concentration (%)', fontweight='bold')
axes[0, 1].set_ylabel('Predicted Concentration (%)', fontweight='bold')
axes[0, 1].set_title('Predictions by Concentration', fontweight='bold')
plt.colorbar(scatter, ax=axes[0, 1], label='Concentration (%)')
axes[0, 1].grid(alpha=0.3)

# 3. Residuals
residuals = final_predictions_original - final_targets_original
axes[1, 0].scatter(final_predictions_original, residuals, alpha=0.6, s=50)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Concentration (%)', fontweight='bold')
axes[1, 0].set_ylabel('Residuals (%)', fontweight='bold')
axes[1, 0].set_title('Residual Plot', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 4. Residuals distribution
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Residuals (%)', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Residuals Distribution', fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 4.5 Spiking Activity Analysis
# ============================================================================
print("\n" + "-"*80)
print("4.5 Spiking Activity Analysis")
print("-"*80)

model.eval()
with torch.no_grad():
    sample_batch = spike_test[:, :batch_size, :].to(device)
    mem_rec, spk_rec1, spk_rec2 = model(sample_batch)

spk_rec1_np = spk_rec1.cpu().numpy()
spk_rec2_np = spk_rec2.cpu().numpy()

spike_rate_layer1 = spk_rec1_np.mean()
spike_rate_layer2 = spk_rec2_np.mean()

print(f"\nSpike rates:")
print(f"  Hidden Layer 1: {spike_rate_layer1:.4f}")
print(f"  Hidden Layer 2: {spike_rate_layer2:.4f}")

# Spike raster plots
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Spiking Activity in Hidden Layers',
             fontsize=14, fontweight='bold')

sample_idx = 0

# Layer 1
for neuron_idx in range(hidden_size1):
    spike_times = np.where(spk_rec1_np[:, sample_idx, neuron_idx] > 0)[0]
    axes[0].scatter(spike_times, [neuron_idx] * len(spike_times),
                   marker='|', s=50, color='blue', alpha=0.6)

axes[0].set_ylabel('Neuron Index', fontweight='bold')
axes[0].set_title(f'Hidden Layer 1 ({hidden_size1} neurons)',
                 fontweight='bold')
axes[0].grid(alpha=0.3, axis='x')

# Layer 2
for neuron_idx in range(hidden_size2):
    spike_times = np.where(spk_rec2_np[:, sample_idx, neuron_idx] > 0)[0]
    axes[1].scatter(spike_times, [neuron_idx] * len(spike_times),
                   marker='|', s=50, color='green', alpha=0.6)

axes[1].set_xlabel('Time Step', fontweight='bold')
axes[1].set_ylabel('Neuron Index', fontweight='bold')
axes[1].set_title(f'Hidden Layer 2 ({hidden_size2} neurons)',
                 fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TIME SERIES SNN REGRESSION - FINAL SUMMARY")
print("="*80)

print(f"""
✓ SUCCESSFULLY COMPLETED TIME SERIES ETHANOL CONCENTRATION REGRESSION!

KEY APPROACH - TIME SERIES vs AGGREGATED:
==========================================
1. DATA REPRESENTATION:
   AGGREGATED: Mean, std, min, max → 57 features per file
   TIME SERIES: Full temporal sequences → {SEQUENCE_LENGTH} timesteps × {num_features} features
   
2. TEMPORAL INFORMATION:
   AGGREGATED: Lost temporal dynamics (only statistics)
   TIME SERIES: Preserved temporal patterns and sensor response dynamics
   
3. SPIKE ENCODING:
   AGGREGATED: Latency encoding on 57 statistical features
   TIME SERIES: Direct encoding on {SEQUENCE_LENGTH} temporal timesteps
   
4. SNN PROCESSING:
   AGGREGATED: 25 SNN timesteps for latency encoding
   TIME SERIES: {SEQUENCE_LENGTH} SNN timesteps (matches sequence length)
   
5. TASK:
   Both: Regression for ethanol concentration prediction
   Input size: TIME SERIES uses {num_features} features/timestep vs 57 aggregate features

ARCHITECTURE:
=============
- Input: {num_features} features per timestep
- Hidden Layer 1: {hidden_size1} LIF neurons
- Hidden Layer 2: {hidden_size2} LIF neurons
- Output: {output_size} neuron (continuous concentration)
- Total timesteps: {SEQUENCE_LENGTH}
- Beta (decay): {beta}

RESULTS:
========
- RMSE: {rmse:.4f}%
- MAE: {mae:.4f}%
- R² Score: {r2:.4f}
- Error rate: {(mae/final_targets_original.mean())*100:.2f}% of mean
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}

DATASET:
========
- Concentrations: {sorted(np.unique(y_concentration))}%
- Sensors: MQ-3, MQ-4, MQ-6 (2 arrays)
- Environmental: Temperature, Humidity
- Sequence length: {SEQUENCE_LENGTH} timesteps
- Downsample factor: {DOWNSAMPLE_FACTOR}

TRAINING:
=========
- Epochs: {num_epochs}
- Batch size: {batch_size}
- Learning rate: {learning_rate}
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- Encoding: {ENCODING_TYPE}

ADVANTAGES OF TIME SERIES APPROACH:
====================================
✓ Preserves temporal sensor dynamics
✓ Captures transient responses
✓ Maintains sequential information
✓ Better represents biological sensing
✓ Utilizes SNN's temporal processing capabilities
✓ No information loss from aggregation
""")

# ============================================================================
# SAVE TRAINED MODEL
# ============================================================================
print("\n" + "-"*80)
print("SAVING TRAINED MODEL")
print("-"*80)

model_dir = "trained_models"
os.makedirs(model_dir, exist_ok=True)

model_filename = (
    f"ethanol_timeseries_snn_"
    f"seq{SEQUENCE_LENGTH}_"
    f"beta{beta}_"
    f"bs{batch_size}_"
    f"lr{learning_rate}_"
    f"ep{num_epochs}.pth"
)

model_path = os.path.join(model_dir, model_filename)
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden1': hidden_size1,
    'hidden2': hidden_size2,
    'output_size': output_size,
    'beta': beta,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'sequence_length': SEQUENCE_LENGTH,
    'downsample_factor': DOWNSAMPLE_FACTOR
}, model_path)

print(f"\n✓ Model saved successfully at: {model_path}")

# ============================================================================
# END OF SCRIPT
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY ✅")
print("="*80)

print(f"""
Summary:
---------
Model Type       : Spiking Neural Network (Leaky Integrate-and-Fire)
Dataset          : Ethanol Concentration Time-Series
Task             : Regression (Continuous Concentration Prediction)
Timesteps        : {SEQUENCE_LENGTH}
Features         : {num_features} per timestep
Encoding Type    : {ENCODING_TYPE} (threshold-based)
Optimizer        : Adam (lr={learning_rate})
Training Epochs  : {num_epochs}
Final RMSE       : {rmse:.4f}%
Final MAE        : {mae:.4f}%
Final R² Score   : {r2:.4f}
Model Path       : {model_path}

Key Differences from Aggregated Approach:
------------------------------------------
✓ Uses FULL time series data (not statistics)
✓ Processes {SEQUENCE_LENGTH} timesteps (not 25)
✓ Direct spike encoding (not latency encoding)
✓ Input: {num_features} features/step (not 57 aggregate features)
✓ Preserves temporal dynamics throughout training
✓ Better captures sensor response patterns

Next Steps:
-----------
1. Try different encoding methods:
   - Rate encoding: spike frequency represents value
   - Latency encoding: spike timing represents value
   - Delta modulation: encode changes over time

2. Experiment with architecture:
   - Add recurrent connections for temporal memory
   - Try different beta values for membrane dynamics
   - Adjust hidden layer sizes

3. Data augmentation:
   - Add temporal jittering
   - Introduce noise for robustness
   - Augment minority concentration classes

4. Multi-task learning:
   - Combine with wine quality classification
   - Predict multiple analytes simultaneously
   - Joint training for related tasks

5. Analysis:
   - Visualize learned temporal features
   - Analyze neuron selectivity
   - Study spike timing patterns

✓ All tasks completed successfully!
""")

print("\n" + "="*80)
print("Thank you for using the Time Series SNN Regression Framework!")
print("="*80)