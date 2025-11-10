"""
Wine Quality Classification using Spiking Neural Networks (SNNs) with Temporal Spike
Attribution (TSA) for Explainability.

This script extends the baseline time-series SNN classifier by integrating Temporal Spike
Attribution following the analytical method from Nguyen et al. (2021).

UPDATED IMPLEMENTATION - Key Changes from Reference Paper:
===========================================================

1. CLASSIFICATION METHOD:
   - Uses summed output MEMBRANE POTENTIALS (not spike counts) for classification
   - Aligns with the reference TSA implementation where decisions are based on 
     accumulated membrane potential over time

2. TSA COMPUTATION METHOD:
   - ANALYTICAL forward-pass method (NOT gradient-based)
   - Computes spike contributions: N(t) = beta^(tc - t) for each spike
   - Forward-propagates contributions through network weights
   - Weights final attribution by output class probabilities
   
3. SPIKE CONTRIBUTION FORMULA:
   - For each spike at time t: contribution = beta^(current_time - t)
   - Beta is the membrane potential decay parameter
   - Earlier spikes have higher contribution due to exponential decay
   
4. ATTRIBUTION PROPAGATION:
   - Layer-wise forward pass through network weights
   - Diagonal matrices of spike contributions multiplied by weight matrices
   - Final scores weighted by softmax probabilities of output classes

Reference: Nguyen, E., Huynh, M., Karadanyan, T., Doan, T., Nowotny, T., & Hassaine, S. (2021). 
Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks. 
arXiv preprint arXiv:2111.00417.

Dataset: Wine Time-Series Dataset (MQ sensor arrays + environmental signals)
Architecture: Feedforward LIF SNN with static fully-connected layers (28, 8 hidden neurons)
Explainability: Analytical temporal spike attribution + spike ablation verification
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import snntorch as snn

warnings.filterwarnings("ignore")

# Reproducibility -----------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_wine_dataset(base_path: str) -> pd.DataFrame:
    """Load raw wine time-series files while preserving temporal order."""
    ts_columns = [
        "Rel_Humidity (%)", "Temperature (C)",
        "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
        "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
    ]

    all_data = []
    print(f"Loading data from: {base_path}")

    for root, _, files in os.walk(base_path):
        folder_name = os.path.basename(root)
        if folder_name.lower() in ["lq_wines", "hq_wines", "aq_wines"]:
            print(f"Processing folder: {folder_name}...")
            for file_name in files:
                if not file_name.endswith(".txt"):
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    df_file = pd.read_csv(file_path, sep=r"\s+", header=None, names=ts_columns)
                    df_file["Filename"] = file_name
                    df_file["Time_Point"] = range(len(df_file))

                    df_file["Quality_Label"] = folder_name.split("_")[0][:2].upper()
                    df_file["Brand"] = file_name[3:9]
                    df_file["Bottle"] = file_name[10:13]
                    rep_start_index = file_name.rfind("_R") + 1
                    df_file["Repetition"] = file_name[rep_start_index:rep_start_index + 3]

                    all_data.append(df_file)
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"Error processing {file_name}: {exc}")

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows loaded: {len(final_df)}")
    return final_df


def remove_stabilisation(df: pd.DataFrame, stabilisation_period: int) -> pd.DataFrame:
    """Drop the initial stabilisation samples per file."""
    cleaned = []
    for filename in df["Filename"].unique():
        file_slice = df[df["Filename"] == filename].copy()
        if len(file_slice) <= stabilisation_period:
            continue
        file_slice = file_slice.iloc[stabilisation_period:].reset_index(drop=True)
        cleaned.append(file_slice)
    if not cleaned:
        raise ValueError("All files were shorter than the stabilisation period")
    return pd.concat(cleaned, ignore_index=True)


def build_fixed_length_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int,
    downsample_factor: int
) -> Tuple[np.ndarray, list, list]:
    """Create fixed-length sequences with optional downsampling."""
    sequences = []
    labels = []
    metadata = []

    for filename in df["Filename"].unique():
        file_data = df[df["Filename"] == filename]
        sequence = file_data[feature_cols].values
        if downsample_factor > 1:
            sequence = sequence[::downsample_factor]

        if len(sequence) < sequence_length:
            padding = np.repeat(sequence[-1:], sequence_length - len(sequence), axis=0)
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]

        sequences.append(sequence.astype(np.float32))
        labels.append(file_data["Quality_Label"].iloc[0])
        metadata.append({
            "Filename": filename,
            "Quality_Label": file_data["Quality_Label"].iloc[0],
            "Brand": file_data["Brand"].iloc[0],
            "Bottle": file_data["Bottle"].iloc[0]
        })

    sequences = np.stack(sequences)
    return sequences, labels, metadata


def clean_and_normalise_sequences(sequences: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Replace invalid values and rescale features to [0, 1]."""
    sequences = sequences.copy()
    num_sequences, num_steps, num_features = sequences.shape

    for idx in range(num_sequences):
        seq = sequences[idx]
        mask_bad = ~np.isfinite(seq)
        if mask_bad.any():
            col_means = np.nanmean(np.where(np.isfinite(seq), seq, np.nan), axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            seq[mask_bad] = np.take(col_means, np.where(mask_bad)[1])
        sequences[idx] = seq

    scaler = MinMaxScaler(feature_range=(0, 1))
    flat = sequences.reshape(-1, num_features)
    flat_scaled = scaler.fit_transform(flat)
    scaled_sequences = flat_scaled.reshape(num_sequences, num_steps, num_features)
    return scaled_sequences.astype(np.float32), scaler


def encode_time_series_direct(sequences: np.ndarray) -> torch.Tensor:
    """Threshold-based encoding (values > 0.5 -> spike)."""
    sequences_tensor = torch.from_numpy(sequences)
    spike_tensor = (sequences_tensor > 0.5).float()
    spike_tensor = spike_tensor.permute(1, 0, 2)  # [timesteps, samples, features]
    return spike_tensor


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class TimeSeriesWineSNN(nn.Module):
    """Feedforward LIF SNN for wine quality classification."""

    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int,
                 output_size: int, beta: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard forward pass returning spike recordings for every layer."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_record_1 = []
        spk_record_2 = []
        spk_record_3 = []
        mem_record_3 = []

        for step in range(x.size(0)):
            x_step = x[step]
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_record_1.append(spk1)
            spk_record_2.append(spk2)
            spk_record_3.append(spk3)
            mem_record_3.append(mem3)

        spk1 = torch.stack(spk_record_1)
        spk2 = torch.stack(spk_record_2)
        spk3 = torch.stack(spk_record_3)
        mem3 = torch.stack(mem_record_3)
        return mem3, spk3, spk1, spk2


# =============================================================================
# TRAINING AND EVALUATION UTILITIES
# =============================================================================

def calculate_logits_from_membrane(mem_record: torch.Tensor) -> torch.Tensor:
    """Aggregate output membrane potentials over time to obtain logits.
    
    Following the reference TSA implementation, classification is performed
    on the sum of membrane potentials rather than spike counts.
    """
    return mem_record.sum(dim=0)


def train_model(
    model: TimeSeriesWineSNN,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    device: torch.device
) -> Dict[str, list]:
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for spike_batch, target_batch in train_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            target_batch = target_batch.to(device)

            mem_rec, spk_rec, _, _ = model(spike_batch)
            logits = calculate_logits_from_membrane(mem_rec)
            loss = criterion(logits, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            running_total += target_batch.size(0)
            running_correct += (preds == target_batch).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * running_correct / running_total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for spike_batch, target_batch in test_loader:
                spike_batch = spike_batch.permute(1, 0, 2).to(device)
                target_batch = target_batch.to(device)

                mem_rec, spk_rec, _, _ = model(spike_batch)
                logits = calculate_logits_from_membrane(mem_rec)
                loss = criterion(logits, target_batch)

                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_total += target_batch.size(0)
                test_correct += (preds == target_batch).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_batch.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = 100.0 * test_correct / test_total
        epoch_test_f1 = f1_score(all_targets, all_preds, average="weighted")

        scheduler.step(epoch_test_loss)

        history["train_loss"].append(epoch_train_loss)
        history["test_loss"].append(epoch_test_loss)
        history["train_acc"].append(epoch_train_acc)
        history["test_acc"].append(epoch_test_acc)
        history["test_f1"].append(epoch_test_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}] | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Test Loss: {epoch_test_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.2f}% | "
                f"Test Acc: {epoch_test_acc:.2f}% | "
                f"F1: {epoch_test_f1:.4f}"
            )

    return history


def predict_with_membrane_potential(model: TimeSeriesWineSNN, sequence: torch.Tensor,
                                    device: torch.device) -> Tuple[int, torch.Tensor]:
    """Make prediction using summed membrane potentials (TSA reference approach)."""
    model.eval()
    with torch.no_grad():
        mem_rec, spk_rec, _, _ = model(sequence.unsqueeze(1).to(device))
        membrane_sums = mem_rec.sum(dim=0)[0].cpu()
    predicted_class = int(torch.argmax(membrane_sums).item())
    return predicted_class, membrane_sums


def extract_spike_times_per_feature(spike_sequence: torch.Tensor) -> list:
    """
    Extract spike times for each input feature from a dense spike tensor.
    
    Args:
        spike_sequence: [timesteps, features] tensor with binary spikes
        
    Returns:
        list of tensors, each containing spike times for one feature
    """
    num_timesteps, num_features = spike_sequence.shape
    spike_trains = []
    
    for feature_idx in range(num_features):
        # Find timesteps where this feature spiked
        spike_times = torch.where(spike_sequence[:, feature_idx] > 0.5)[0].float()
        spike_trains.append(spike_times)
    
    return spike_trains


def spike_contribution(spike_times: torch.Tensor, tc: int, beta: float) -> torch.Tensor:
    """
    Calculate spike contribution N(t) for spikes at given times.
    
    Following the TSA paper: N(t) = beta^(tc - t) for each spike at time t
    
    Args:
        spike_times: Tensor of spike times for a single neuron
        tc: Current time (attribution computed up to this time)
        beta: Membrane potential decay parameter
        
    Returns:
        Tensor of contribution scores for each spike
    """
    if len(spike_times) == 0:
        return torch.tensor([0.0])
    
    diff_t = tc - spike_times
    score = beta ** diff_t
    return score


def compute_temporal_spike_attribution_analytical(
    model: TimeSeriesWineSNN,
    sequence: torch.Tensor,
    target_class: int,
    beta: float,
    device: torch.device
) -> torch.Tensor:
    """
    Analytical TSA following the reference implementation.
    
    This method computes attribution by:
    1. Extracting spike times from the input sequence
    2. Computing spike contributions N(t) = beta^(tc - t)
    3. Forward-propagating through network weights
    4. Weighting by output class probability
    
    Args:
        model: The trained SNN model
        sequence: Input spike sequence [timesteps, features]
        target_class: Class to compute attribution for
        beta: Membrane decay parameter
        device: Computation device
        
    Returns:
        Attribution map [timesteps, features]
    """
    model.eval()
    sequence = sequence.to(device)
    
    num_timesteps, num_features = sequence.shape
    
    # Get model predictions and internal states
    with torch.no_grad():
        mem_rec, spk_rec, spk_rec1, spk_rec2 = model(sequence.unsqueeze(1))
        
        # Compute output probabilities from membrane potentials
        mem_sum = mem_rec.sum(dim=0)[0]  # Sum over time
        probs = torch.softmax(mem_sum, dim=0)
    
    # Extract spike trains for input and hidden layers
    input_spike_trains = extract_spike_times_per_feature(sequence)
    
    # Extract hidden layer spike trains
    hidden1_spikes = spk_rec1[:, 0, :].cpu()  # [timesteps, hidden1_size]
    hidden2_spikes = spk_rec2[:, 0, :].cpu()  # [timesteps, hidden2_size]
    
    hidden1_spike_trains = extract_spike_times_per_feature(hidden1_spikes)
    hidden2_spike_trains = extract_spike_times_per_feature(hidden2_spikes)
    
    # Get network weights
    with torch.no_grad():
        w1 = model.fc1.weight.t().cpu()  # [input_size, hidden1_size]
        w2 = model.fc2.weight.t().cpu()  # [hidden1_size, hidden2_size]
        w3 = model.fc3.weight.t().cpu()  # [hidden2_size, output_size]
    
    # Initialize attribution map
    attribution_map = torch.zeros(num_timesteps, num_features)
    
    # Compute attribution for each timestep
    tc = num_timesteps
    
    for t in range(num_timesteps):
        # Layer 0: Input layer spike contributions
        layer0_contrib = []
        for feat_idx in range(num_features):
            spk_times_at_t = input_spike_trains[feat_idx]
            spk_times_at_t = spk_times_at_t[spk_times_at_t == t]
            contrib = spike_contribution(spk_times_at_t, tc, beta)
            layer0_contrib.append(contrib.sum())
        
        layer0_contrib = torch.tensor(layer0_contrib)
        layer0_contrib_diag = torch.diag(layer0_contrib)
        
        # Forward through first weight matrix
        layer0_weighted = torch.matmul(layer0_contrib_diag, w1)
        
        # Layer 1: Hidden layer 1 spike contributions
        hidden1_size = w2.shape[0]
        layer1_contrib = []
        for h_idx in range(hidden1_size):
            spk_times_at_t = hidden1_spike_trains[h_idx]
            spk_times_at_t = spk_times_at_t[spk_times_at_t == t]
            contrib = spike_contribution(spk_times_at_t, tc, beta)
            layer1_contrib.append(contrib.sum())
        
        layer1_contrib = torch.tensor(layer1_contrib)
        layer1_contrib_diag = torch.diag(layer1_contrib)
        layer1_weighted = torch.matmul(layer1_contrib_diag, w2)
        
        # Layer 2: Hidden layer 2 spike contributions
        hidden2_size = w3.shape[0]
        layer2_contrib = []
        for h_idx in range(hidden2_size):
            spk_times_at_t = hidden2_spike_trains[h_idx]
            spk_times_at_t = spk_times_at_t[spk_times_at_t == t]
            contrib = spike_contribution(spk_times_at_t, tc, beta)
            layer2_contrib.append(contrib.sum())
        
        layer2_contrib = torch.tensor(layer2_contrib)
        layer2_contrib_diag = torch.diag(layer2_contrib)
        layer2_weighted = torch.matmul(layer2_contrib_diag, w3)
        
        # Forward propagate contributions
        w_spk_contr_t = torch.matmul(layer0_weighted, layer1_weighted)
        w_spk_contr_t = torch.matmul(w_spk_contr_t, layer2_weighted)
        
        # Weight by target class probability
        class_probs_diag = torch.diag(probs.cpu())
        scores = torch.matmul(w_spk_contr_t, class_probs_diag)
        
        # Extract attribution for target class
        attribution_map[t, :] = scores[:, target_class]
    
    return attribution_map


def plot_attribution_heatmap(attribution: torch.Tensor, feature_names: list,
                             title: str = "Temporal Spike Attribution",
                             cmap: str = "coolwarm") -> None:
    """Render a heatmap showing spike importance over time."""
    attr_np = attribution.numpy()
    plt.figure(figsize=(14, 6))
    sns.heatmap(attr_np.T, cmap=cmap, center=0.0,
                xticklabels=50, yticklabels=feature_names)
    plt.xlabel("Time step")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def summarise_top_spikes(attribution: torch.Tensor, feature_names: list,
                         top_k: int = 20) -> pd.DataFrame:
    """Return a table of the most influential spikes."""
    flat_attr = attribution.view(-1)
    abs_attr = flat_attr.abs()
    top_k = min(top_k, flat_attr.numel())
    values, indices = torch.topk(abs_attr, k=top_k)

    num_features = len(feature_names)
    rows = []
    for value, index in zip(values, indices):
        timestep = int(index.item() // num_features)
        feature_idx = int(index.item() % num_features)
        rows.append({
            "timestep": timestep,
            "feature": feature_names[feature_idx],
            "attribution": float(flat_attr[index].item())
        })
    return pd.DataFrame(rows)


def verify_attribution_by_spike_editing(
    model: TimeSeriesWineSNN,
    original_sequence: torch.Tensor,
    attribution: torch.Tensor,
    device: torch.device,
    top_k: int,
    random_trials: int = 5
) -> Dict[str, object]:
    """Flip top-attributed spikes and compare prediction changes."""
    model.eval()
    num_features = original_sequence.shape[1]

    importance = attribution.abs()  # Use absolute values for ranking
    spike_mask = original_sequence.view(-1) > 0.5
    available_indices = torch.nonzero(spike_mask, as_tuple=False).squeeze(-1)

    if available_indices.numel() == 0:
        raise ValueError("No active spikes available for attribution verification")

    flat_importance = importance.view(-1)
    imp_values = flat_importance[available_indices]
    if imp_values.numel() == 0:
        raise ValueError("No positively attributed spikes found for verification")

    top_k = min(top_k, imp_values.numel())
    top_order = torch.topk(imp_values, k=top_k).indices
    top_indices = available_indices[top_order]

    edited_sequence = original_sequence.clone()
    for idx in top_indices:
        timestep = int(idx.item() // num_features)
        feature_idx = int(idx.item() % num_features)
        edited_sequence[timestep, feature_idx] = 0.0

    baseline_pred, baseline_mem = predict_with_membrane_potential(model, original_sequence, device)
    edited_pred, edited_mem = predict_with_membrane_potential(model, edited_sequence, device)

    random_outcomes = []
    rng = torch.Generator().manual_seed(42)
    for _ in range(random_trials):
        random_indices = available_indices[torch.randperm(available_indices.numel(), generator=rng)[:top_k]]
        random_sequence = original_sequence.clone()
        for idx in random_indices:
            timestep = int(idx.item() // num_features)
            feature_idx = int(idx.item() % num_features)
            random_sequence[timestep, feature_idx] = 0.0
        rand_pred, rand_mem = predict_with_membrane_potential(model, random_sequence, device)
        random_outcomes.append({
            "prediction": rand_pred,
            "membrane_sums": rand_mem
        })

    return {
        "baseline_prediction": baseline_pred,
        "baseline_membrane": baseline_mem,
        "edited_prediction": edited_pred,
        "edited_membrane": edited_mem,
        "random_outcomes": random_outcomes,
        "top_indices": top_indices.cpu()
    }


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

print("\n" + "=" * 80)
print("PART 1: LOADING WINE TIME-SERIES DATASET")
print("=" * 80)

dataset_path = "data/wine"
if not os.path.exists(dataset_path):
    dataset_path = "Dataset"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found")

raw_df = load_wine_dataset(dataset_path)
if raw_df.empty:
    raise ValueError("Dataset is empty")

print(f"\nDataset shape: {raw_df.shape}")
print(f"Unique files: {raw_df['Filename'].nunique()}")

print("\n" + "-" * 80)
print("1.1 Removing Stabilisation Period")
print("-" * 80)

STABILISATION_PERIOD = 500
clean_df = remove_stabilisation(raw_df, STABILISATION_PERIOD)
print(f"\nDataset shape after removing stabilisation: {clean_df.shape}")

print("\n" + "-" * 80)
print("1.2 Preparing Time Series Sequences")
print("-" * 80)

sensor_cols = [
    "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
    "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
]
environmental_cols = ["Rel_Humidity (%)", "Temperature (C)"]
feature_cols = sensor_cols + environmental_cols

SEQUENCE_LENGTH = 500
DOWNSAMPLE_FACTOR = 2

sequences, labels, metadata = build_fixed_length_sequences(
    clean_df,
    feature_cols,
    sequence_length=SEQUENCE_LENGTH,
    downsample_factor=DOWNSAMPLE_FACTOR
)

print(f"\nTotal sequences: {len(sequences)}")
print(f"Sequence shape: {sequences[0].shape} (timesteps, features)")

print("\n" + "-" * 80)
print("1.3 Cleaning and Normalising Time Series")
print("-" * 80)

normalised_sequences, scaler = clean_and_normalise_sequences(sequences)
flat_normalised = normalised_sequences.reshape(-1, normalised_sequences.shape[-1])
print(f"\nNormalised range: [{flat_normalised.min():.3f}, {flat_normalised.max():.3f}]")

print("\n" + "-" * 80)
print("1.4 Encoding Labels")
print("-" * 80)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
label_distribution = {
    label: int(np.sum(y_encoded == encoded))
    for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}
print("\nLabel distribution:")
for label, count in label_distribution.items():
    print(f"  {label}: {count} samples")

print("\n" + "-" * 80)
print("1.5 Train-Test Split")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    normalised_sequences,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\n" + "=" * 80)
print("PART 2: SPIKE ENCODING")
print("=" * 80)

spike_train = encode_time_series_direct(X_train)
spike_test = encode_time_series_direct(X_test)

print(f"Training spike tensor shape: {spike_train.shape}")
print(f"Test spike tensor shape: {spike_test.shape}")

print("\n" + "=" * 80)
print("PART 3: MODEL SETUP")
print("=" * 80)

input_size = len(feature_cols)
hidden_size1 = 28
hidden_size2 = 8
output_size = len(label_encoder.classes_)
beta = 0.9

model = TimeSeriesWineSNN(
    input_size=input_size,
    hidden_size1=hidden_size1,
    hidden_size2=hidden_size2,
    output_size=output_size,
    beta=beta
).to(device)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
num_epochs = 50
batch_size = 16

train_dataset = torch.utils.data.TensorDataset(
    spike_train.permute(1, 0, 2), torch.from_numpy(y_train)
)
test_dataset = torch.utils.data.TensorDataset(
    spike_test.permute(1, 0, 2), torch.from_numpy(y_test)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

print("\n" + "=" * 80)
print("PART 4: TRAINING")
print("=" * 80)

history = train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=num_epochs,
    device=device
)

print("\nTraining complete!")

print("\n" + "-" * 80)
print("4.1 Training Curves")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(history["test_loss"], label="Test Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title("Loss")

axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
axes[1].plot(history["test_acc"], label="Test Acc", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_title("Accuracy")

axes[2].plot(history["test_f1"], label="Test F1", linewidth=2, color="green")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("F1 Score")
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_title("F1 Score")

plt.tight_layout()
plt.show()

print("\n" + "-" * 80)
print("4.2 Final Evaluation")
print("-" * 80)

model.eval()
final_predictions = []
final_targets = []

with torch.no_grad():
    for spike_batch, target_batch in test_loader:
        spike_batch = spike_batch.permute(1, 0, 2).to(device)
        mem_rec, _, _, _ = model(spike_batch)
        logits = calculate_logits_from_membrane(mem_rec)
        preds = torch.argmax(logits, dim=1)
        final_predictions.extend(preds.cpu().numpy())
        final_targets.extend(target_batch.numpy())

final_predictions = np.array(final_predictions)
final_targets = np.array(final_targets)

accuracy = accuracy_score(final_targets, final_predictions)
f1 = f1_score(final_targets, final_predictions, average="weighted")

print("\n" + "=" * 80)
print("FINAL PERFORMANCE")
print("=" * 80)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(final_targets, final_predictions,
                            target_names=label_encoder.classes_))

cm = confusion_matrix(final_targets, final_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\n" + "-" * 80)
print("4.3 Hidden Layer Spiking Activity")
print("-" * 80)

model.eval()
with torch.no_grad():
    sample_batch = spike_test[:, :batch_size, :].to(device)
    _, spk_rec3, spk_rec1, spk_rec2 = model(sample_batch)

spike_rate_layer1 = spk_rec1.cpu().numpy().mean()
spike_rate_layer2 = spk_rec2.cpu().numpy().mean()
print(f"Hidden Layer 1 spike rate: {spike_rate_layer1:.4f}")
print(f"Hidden Layer 2 spike rate: {spike_rate_layer2:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Hidden Layer Spike Rasters", fontsize=14)

sample_idx = 0
spk_rec1_np = spk_rec1.cpu().numpy()
spk_rec2_np = spk_rec2.cpu().numpy()

for neuron_idx in range(hidden_size1):
    spike_times = np.where(spk_rec1_np[:, sample_idx, neuron_idx] > 0)[0]
    axes[0].scatter(spike_times, [neuron_idx] * len(spike_times), marker="|", s=50, color="blue", alpha=0.6)
axes[0].set_ylabel("Neuron")
axes[0].set_title(f"Hidden Layer 1 ({hidden_size1} neurons)")
axes[0].grid(alpha=0.3, axis="x")

for neuron_idx in range(hidden_size2):
    spike_times = np.where(spk_rec2_np[:, sample_idx, neuron_idx] > 0)[0]
    axes[1].scatter(spike_times, [neuron_idx] * len(spike_times), marker="|", s=50, color="green", alpha=0.6)
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Neuron")
axes[1].set_title(f"Hidden Layer 2 ({hidden_size2} neurons)")
axes[1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

# =============================================================================
# PART 5: TEMPORAL SPIKE ATTRIBUTION (ANALYTICAL METHOD)
# =============================================================================

print("\n" + "=" * 80)
print("PART 5: TEMPORAL SPIKE ATTRIBUTION")
print("=" * 80)

print("""
TSA METHODOLOGY (Following Reference Implementation):
======================================================

The Temporal Spike Attribution method used here is ANALYTICAL, not gradient-based.

Key Steps:
1. Extract spike times from input and all hidden layers
2. For each timestep t, compute spike contribution: N(t) = beta^(tc - t)
   - tc = current time (total sequence length)
   - beta = membrane decay parameter (0.9)
   - Earlier spikes have larger contribution due to exponential decay

3. Create diagonal matrices of spike contributions per layer
4. Forward-propagate through network weights:
   - Layer 0 (input): N_0(t) * W_1
   - Layer 1 (hidden1): N_1(t) * W_2  
   - Layer 2 (hidden2): N_2(t) * W_3
   - Combine: attribution(t) = N_0 @ W_1 @ N_1 @ W_2 @ N_2 @ W_3

5. Weight by output class probabilities (from softmax of membrane potentials)
6. Result: [timesteps × features] attribution map showing importance of each spike

This differs from gradient-based methods:
- NO backpropagation through the network
- NO noise injection or smoothing
- Deterministic, analytical computation
- Based on spike timing and weight propagation
""")

# Identify a correctly classified test sample for explanation
sample_index = None
model.eval()
for idx in range(spike_test.shape[1]):
    sample_sequence = spike_test[:, idx, :].clone()
    pred_class, _ = predict_with_membrane_potential(model, sample_sequence, device)
    if pred_class == y_test[idx]:
        sample_index = idx
        break

if sample_index is None:
    raise RuntimeError("No correctly classified test sample found for attribution")

print(f"Selected sample index: {sample_index}")
print(f"True label: {label_encoder.inverse_transform([y_test[sample_index]])[0]}")

sample_spike_sequence = spike_test[:, sample_index, :].clone()
true_label = y_test[sample_index]
pred_label, membrane_sums = predict_with_membrane_potential(model, sample_spike_sequence, device)

print(f"Predicted label: {label_encoder.inverse_transform([pred_label])[0]}")
print(f"Output membrane sums: {membrane_sums.numpy()}")

attribution = compute_temporal_spike_attribution_analytical(
    model,
    sample_spike_sequence,
    target_class=pred_label,
    beta=beta,
    device=device
)

print("\nTop attributed spikes:")
print(summarise_top_spikes(attribution, feature_cols, top_k=10))

plot_attribution_heatmap(attribution, feature_cols)

verification = verify_attribution_by_spike_editing(
    model,
    sample_spike_sequence,
    attribution,
    device=device,
    top_k=50,
    random_trials=3
)

print("\nVerification results:")
print(f"  Baseline prediction: {label_encoder.inverse_transform([verification['baseline_prediction']])[0]}")
print(f"  Edited prediction:   {label_encoder.inverse_transform([verification['edited_prediction']])[0]}")
print(f"  Baseline membrane sums: {verification['baseline_membrane'].numpy()}")
print(f"  Edited membrane sums:   {verification['edited_membrane'].numpy()}")

for idx, random_outcome in enumerate(verification["random_outcomes"], start=1):
    print(
        f"  Random edit #{idx}: prediction = "
        f"{label_encoder.inverse_transform([random_outcome['prediction']])[0]}, "
        f"membrane sums = {random_outcome['membrane_sums'].numpy()}"
    )

# =============================================================================
# SAVE TRAINED MODEL
# =============================================================================

model_dir = "trained_models"
os.makedirs(model_dir, exist_ok=True)
model_filename = (
    f"wine_timeseries_snn_tsa_seq{SEQUENCE_LENGTH}_"
    f"beta{beta}_bs{batch_size}_lr0.001_ep{num_epochs}.pth"
)
model_path = os.path.join(model_dir, model_filename)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": input_size,
    "hidden1": hidden_size1,
    "hidden2": hidden_size2,
    "output_size": output_size,
    "beta": beta,
    "scaler": scaler,
    "label_encoder": label_encoder.classes_,
    "feature_cols": feature_cols
}, model_path)

print(f"\nModel saved to: {model_path}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
