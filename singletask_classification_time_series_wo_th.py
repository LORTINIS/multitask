"""
Wine Quality Classification using Spiking Neural Networks (SNNs) with snntorch
VARIANT: Gas-sensor-only time series (no temperature / humidity channels)

This script implements a single-task SNN for wine quality classification with:
1. Time series data loading and preprocessing (6 gas sensor channels)
2. Direct spike encoding from temporal sequences
3. SNN architecture with temporal processing
4. Model training and evaluation for wine quality classification

Dataset: Wine Time-Series Dataset
Architecture: LIF-based SNN with 2 hidden layers (28, 8 neurons)
Modified for: Time series input processing without temperature/humidity
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

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
print("PART 1: LOADING WINE TIME-SERIES DATASET (GAS SENSORS ONLY)")
print("="*80)

def load_wine_dataset(base_path):
    """
    Loads the wine time-series dataset preserving temporal structure.
    """
    all_data = []

    ts_columns = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]

    print(f"Loading data from: {base_path}")

    for root, _, files in os.walk(base_path):
        folder_name = os.path.basename(root)

        if folder_name.lower() in ['lq_wines', 'hq_wines', 'aq_wines']:
            print(f"Processing folder: {folder_name}...")

            for file_name in files:
                if file_name.endswith('.txt'):
                    file_path = os.path.join(root, file_name)

                    try:
                        df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=ts_columns)
                        df_file['Filename'] = file_name
                        df_file['Time_Point'] = range(len(df_file))

                        # Extract quality label
                        df_file['Quality_Label'] = folder_name.split('_')[0][:2].upper()
                        df_file['Brand'] = file_name[3:9]
                        df_file['Bottle'] = file_name[10:13]

                        rep_start_index = file_name.rfind('_R') + 1
                        df_file['Repetition'] = file_name[rep_start_index:rep_start_index + 3]

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

wine_df = load_wine_dataset(dataset_path)

if wine_df.empty:
    raise ValueError("Dataset is empty")

print(f"\nDataset shape: {wine_df.shape}")
print(f"Unique files: {wine_df['Filename'].nunique()}")

# ============================================================================
# 1.1 Remove Stabilization Period
# ============================================================================
print("\n" + "-"*80)
print("1.1 Removing Stabilization Period")
print("-"*80)

stabilization_period = 500
processed_files = []

for filename in wine_df['Filename'].unique():
    file_data = wine_df[wine_df['Filename'] == filename].copy()

    if len(file_data) > stabilization_period:
        file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
        processed_files.append(file_data)

wine_df = pd.concat(processed_files, ignore_index=True)
print(f"\nDataset shape after removing stabilization: {wine_df.shape}")

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
feature_cols = sensor_cols

# Configuration
SEQUENCE_LENGTH = 500
DOWNSAMPLE_FACTOR = 2

print(f"\nTime series configuration:")
print(f"  Sequence length: {SEQUENCE_LENGTH}")
print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
print(f"  Features: {len(feature_cols)} (gas sensors only)")

sequences = []
labels = []
metadata = []

for filename in wine_df['Filename'].unique():
    file_data = wine_df[wine_df['Filename'] == filename]

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
    labels.append(file_data['Quality_Label'].iloc[0])

    metadata.append({
        'Filename': filename,
        'Quality_Label': file_data['Quality_Label'].iloc[0],
        'Brand': file_data['Brand'].iloc[0],
        'Bottle': file_data['Bottle'].iloc[0]
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
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data)

normalized_sequences = []
for seq in sequences:
    normalized_seq = scaler.transform(seq)
    normalized_sequences.append(normalized_seq)

all_normalized = np.vstack(normalized_sequences)
print(f"\nNormalized range: [{all_normalized.min():.3f}, {all_normalized.max():.3f}]")

# ============================================================================
# 1.4 Prepare Labels
# ============================================================================
print("\n" + "-"*80)
print("1.4 Encoding Labels")
print("-"*80)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

print(f"\nLabel distribution:")
for label, encoded in zip(label_encoder.classes_,
                          label_encoder.transform(label_encoder.classes_)):
    count = np.sum(y_encoded == encoded)
    print(f"  {label}: {count} samples (encoded as {encoded})")

# ============================================================================
# 1.5 Pad Sequences and Create Arrays
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
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print(f"\nTraining label distribution:")
for label, encoded in zip(label_encoder.classes_,
                          label_encoder.transform(label_encoder.classes_)):
    count = np.sum(y_train == encoded)
    print(f"  {label}: {count} samples")

# ============================================================================
# PART 2: SPIKE ENCODING FROM TIME SERIES
# ============================================================================

print("\n" + "="*80)
print("PART 2: SPIKE ENCODING TIME SERIES DATA")
print("="*80)

print("""
ENCODING STRATEGY: DIRECT TIME SERIES TO SPIKES
------------------------------------------------
Instead of using latency encoding on aggregated features, we:
1. Use the time series directly
2. Convert normalized values to binary spikes (threshold-based)
3. Preserve temporal dynamics

Each timestep in the original sequence becomes one SNN timestep.
Values above threshold (0.5) → spike (1), below → no spike (0)
""")

ENCODING_TYPE = 'direct'  # Options: 'direct', 'rate', 'latency'

print(f"\nEncoding type: {ENCODING_TYPE}")

def encode_time_series_direct(sequences):
    """
    Encode time series using direct threshold-based conversion.

    Args:
        sequences: numpy array of shape [samples, timesteps, features]

    Returns:
        spike_data: torch tensor of shape [timesteps, samples, features]
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
    print(f"  Sparsity: {1 - spike_rate:.4f}")

    return spike_data

# Encode training and test data
print("\nEncoding training data:")
spike_train = encode_time_series_direct(X_train)

print("\nEncoding test data:")
spike_test = encode_time_series_direct(X_test)

# Convert labels to tensors
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

print("\n✓ Spike encoding completed!")

# ============================================================================
# PART 3: SNN ARCHITECTURE FOR TIME SERIES
# ============================================================================

print("\n" + "="*80)
print("PART 3: DESIGNING SNN ARCHITECTURE FOR TIME SERIES")
print("="*80)

class TimeSeriesWineSNN(nn.Module):
    """
    Spiking Neural Network for Wine Quality Classification from Time Series

    Architecture:
    - Input: [timesteps, batch, features] time series
    - Hidden Layer 1: 28 LIF neurons
    - Hidden Layer 2: 8 LIF neurons
    - Output Layer: 3 neurons (HQ, LQ, AQ)
    """

    def __init__(self, input_size=6, hidden_size1=28, hidden_size2=8,
                 output_size=3, beta=0.9):
        super(TimeSeriesWineSNN, self).__init__()

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
            mem_rec3: Membrane potential recordings
            spk_rec3: Output spike recordings
            spk_rec1, spk_rec2: Hidden layer spike recordings
        """
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record activity
        spk_rec1 = []
        spk_rec2 = []
        spk_rec3 = []
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

            # Layer 3
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Record
            spk_rec1.append(spk1)
            spk_rec2.append(spk2)
            spk_rec3.append(spk3)
            mem_rec3.append(mem3)

        # Stack recordings
        spk_rec1 = torch.stack(spk_rec1)
        spk_rec2 = torch.stack(spk_rec2)
        spk_rec3 = torch.stack(spk_rec3)
        mem_rec3 = torch.stack(mem_rec3)

        return mem_rec3, spk_rec3, spk_rec1, spk_rec2

# ============================================================================
# 3.1 Instantiate Model
# ============================================================================
print("\n" + "-"*80)
print("3.1 Model Instantiation")
print("-"*80)

input_size = num_features  # 6 features
hidden_size1 = 28
hidden_size2 = 8
output_size = 3
beta = 0.9

print(f"\nModel configuration:")
print(f"  Input size: {input_size} (features per timestep)")
print(f"  Hidden layer 1: {hidden_size1} neurons")
print(f"  Hidden layer 2: {hidden_size2} neurons")
print(f"  Output size: {output_size} classes")
print(f"  Beta: {beta}")
print(f"  Timesteps: {SEQUENCE_LENGTH}")

model = TimeSeriesWineSNN(
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
print("PART 4: TRAINING TIME SERIES SNN")
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
criterion = nn.CrossEntropyLoss()
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
train_accuracies = []
test_accuracies = []
test_f1_scores = []

print("\nStarting training...")
print("-" * 80)

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for spike_batch, target_batch in train_loader:
        spike_batch = spike_batch.permute(1, 0, 2).to(device)
        target_batch = target_batch.to(device)

        mem_rec, spk_rec, _, _ = model(spike_batch)
        logits = mem_rec.sum(dim=0)

        loss = criterion(logits, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        train_total += target_batch.size(0)
        train_correct += (predicted == target_batch).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for spike_batch, target_batch in test_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            target_batch = target_batch.to(device)

            mem_rec, _, _, _ = model(spike_batch)
            logits = mem_rec.sum(dim=0)

            loss = criterion(logits, target_batch)
            test_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            test_total += target_batch.size(0)
            test_correct += (predicted == target_batch).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target_batch.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    f1 = f1_score(all_targets, all_predictions, average='weighted')
    test_f1_scores.append(f1)

    scheduler.step(test_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Test Acc: {test_accuracy:.2f}% | "
              f"F1: {f1:.4f}")

print("\n✓ Training completed!")

# ============================================================================
# 4.2 Visualize Training History
# ============================================================================
print("\n" + "-"*80)
print("4.2 Training History")
print("-"*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(test_losses, label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontweight='bold')
axes[0].set_ylabel('Loss', fontweight='bold')
axes[0].set_title('Training Progress', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(train_accuracies, label='Train Accuracy', linewidth=2)
axes[1].plot(test_accuracies, label='Test Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
axes[1].set_title('Accuracy Over Time', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# F1 Score
axes[2].plot(test_f1_scores, label='Test F1', linewidth=2, color='green')
axes[2].set_xlabel('Epoch', fontweight='bold')
axes[2].set_ylabel('F1 Score', fontweight='bold')
axes[2].set_title('F1 Score', fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

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

        mem_rec, _, _, _ = model(spike_batch)
        logits = mem_rec.sum(dim=0)

        _, predicted = torch.max(logits, 1)

        final_predictions.extend(predicted.cpu().numpy())
        final_targets.extend(target_batch.numpy())

final_predictions = np.array(final_predictions)
final_targets = np.array(final_targets)

accuracy = accuracy_score(final_targets, final_predictions)
f1 = f1_score(final_targets, final_predictions, average='weighted')

print("\n" + "="*80)
print("FINAL PERFORMANCE")
print("="*80)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1:.4f}")

print("\n\nClassification Report:")
print("-" * 80)
print(classification_report(final_targets, final_predictions,
                          target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(final_targets, final_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Time Series Wine Classification (Sensors Only)',
         fontweight='bold', fontsize=14)
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 4.4 Visualize Spiking Activity
# ============================================================================
print("\n" + "-"*80)
print("4.4 Spiking Activity Analysis")
print("-"*80)

model.eval()
with torch.no_grad():
    sample_batch = spike_test[:, :batch_size, :].to(device)
    mem_rec, spk_rec3, spk_rec1, spk_rec2 = model(sample_batch)

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
print("TIME SERIES SNN (SENSORS ONLY) - FINAL SUMMARY")
print("="*80)

print(f"""
✓ COMPLETED TIME SERIES WINE CLASSIFICATION USING GAS SENSORS ONLY!

KEY DIFFERENCES FROM FULL FEATURE SET:
======================================
1. DATA REPRESENTATION:
   - Using 6 MQ sensor channels per timestep (temperature/humidity removed)
   - {SEQUENCE_LENGTH} timesteps per sample preserved

2. SPIKE ENCODING:
   - Direct threshold encoding on sensor-only sequences

3. SNN PROCESSING:
   - {SEQUENCE_LENGTH} SNN timesteps with input size {num_features}

RESULTS:
========
- Final Accuracy: {accuracy*100:.2f}%
- F1 Score: {f1:.4f}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Hidden layers: [{hidden_size1}, {hidden_size2}]
- Output classes: {output_size}
- Encoding type: {ENCODING_TYPE}
- Beta: {beta}
- Learning rate: {learning_rate}
- Epochs: {num_epochs}
- Batch size: {batch_size}
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
    f"wine_timeseries_snn_wo_th_"
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
    'scaler': scaler,
    'label_encoder': label_encoder.classes_
}, model_path)

print(f"\n✓ Model saved successfully at: {model_path}")

# ============================================================================
# END OF SCRIPT
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY ✅ (SENSORS ONLY)")
print("="*80)

print(f"""
Summary:
---------
Model Type       : Spiking Neural Network (Leaky Integrate-and-Fire)
Dataset          : Wine Quality Time-Series (gas sensors)
Timesteps        : {SEQUENCE_LENGTH}
Features         : {num_features}
Encoding Type    : {ENCODING_TYPE}
Optimizer        : Adam (lr={learning_rate})
Training Epochs  : {num_epochs}
Final Accuracy   : {accuracy*100:.2f}%
Final F1 Score   : {f1:.4f}
Model Path       : {model_path}

Next Steps:
-----------
- Compare against full-feature model including environmental sensors
- Evaluate robustness across different downsampling factors
- Explore recurrent SNN variants for longer temporal dependencies
- Investigate explainability differences using TSA on sensor-only data

✓ All tasks completed.
""")
