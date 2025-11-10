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


def compute_N_from_spike_times(spike_times_list, t_prime, beta, fan_in):
    """
    spike_times_list: list of spike times for a single neuron
    t_prime: current explanation time
    beta: decay constant of the LIF neuron
    fan_in: number of incoming connections for this layer (for TSANS)
    Returns: scalar N_i(t')
    """
    N = 0.0
    # TSAS spikes
    for t_k in spike_times_list:
        if t_k <= t_prime:
            N += beta ** (t_prime - t_k)

    # TSANS negative term (absent spikes)
    # one negative contribution per timestep without spikes up to t_prime
    if fan_in is not None:
        # count timesteps with no spikes
        # (t_prime + 1) total timesteps from 0..t_prime inclusive
        spike_set = set(spike_times_list)
        for t_k in range(t_prime + 1):
            if t_k not in spike_set:
                N -= (1.0 / fan_in) * (beta ** (t_prime - t_k))

    return N



def compute_temporal_spike_attribution_analytical(model, x, beta, device):
    x = x.to(device)
    model.eval()

    with torch.no_grad():
        # [T, F] -> model expects [T, 1, F]
        mem_rec, spk_out, hidden_spk1, hidden_spk2 = model(x.unsqueeze(1))

    num_timesteps = mem_rec.shape[0]
    num_inputs = x.shape[1]
    num_hidden1 = hidden_spk1.shape[2]
    num_hidden2 = hidden_spk2.shape[2]
    num_outputs = mem_rec.shape[2]

    # PyTorch Linear: weight shape = (out_features, in_features)
    # For TSA (left-to-right influence): use the TRANSPOSES
    W1_T = model.fc1.weight.data.clone().to(device).t()  # (input_dim, hidden1)
    W2_T = model.fc2.weight.data.clone().to(device).t()  # (hidden1, hidden2)
    W3_T = model.fc3.weight.data.clone().to(device).t()  # (hidden2, output_dim)

    # Extract spike times per neuron
    input_spike_times = [torch.where(x[:, i] > 0)[0].tolist() for i in range(num_inputs)]
    h1_spike_times = [torch.where(hidden_spk1[:, 0, i] > 0)[0].tolist() for i in range(num_hidden1)]
    h2_spike_times = [torch.where(hidden_spk2[:, 0, i] > 0)[0].tolist() for i in range(num_hidden2)]
    out_mem = mem_rec.squeeze(1)  # (T, num_outputs)

    attributions = []

    # fan_in values for TSANS term
    fan_in_1 = num_inputs
    fan_in_2 = num_hidden1
    fan_in_3 = num_hidden2

    for t_prime in range(num_timesteps):
        # Per-time softmax over output membranes
        P_t = torch.softmax(out_mem[t_prime], dim=0)                # (output_dim,)
        P_diag = torch.diag(P_t)                                    # (output_dim, output_dim)

        # Build N^(l)(t') for each layer (sum of decays up to t')
        N0 = torch.zeros(num_inputs, device=device)
        for i in range(num_inputs):
            N0[i] = compute_N_from_spike_times(input_spike_times[i], t_prime, beta, fan_in_1)

        N1 = torch.zeros(num_hidden1, device=device)
        for i in range(num_hidden1):
            N1[i] = compute_N_from_spike_times(h1_spike_times[i], t_prime, beta, fan_in_2)

        N2 = torch.zeros(num_hidden2, device=device)
        for i in range(num_hidden2):
            N2[i] = compute_N_from_spike_times(h2_spike_times[i], t_prime, beta, fan_in_3)

        # Diagonal matrices
        D0 = torch.diag(N0)    # (input_dim, input_dim)
        D1 = torch.diag(N1)    # (hidden1, hidden1)
        D2 = torch.diag(N2)    # (hidden2, hidden2)

        # Algorithm 1: input -> hidden1 -> hidden2 -> output (with weight transposes)
        # Shapes: (in,in)@(in,h1)@(h1,h1)@(h1,h2)@(h2,h2)@(h2,out)@(out,out) = (in,out)
        CI_t = (((((D0 @ W1_T) @ D1) @ W2_T) @ D2) @ W3_T) @ P_diag

        attributions.append(CI_t)

    return torch.stack(attributions)  # (T, input_dim, output_dim)



def plot_attribution_heatmap(attribution: torch.Tensor, feature_names: list,
                             title: str = "Temporal Spike Attribution",
                             cmap: str = "coolwarm") -> None:
    """Render a heatmap showing spike importance over time."""
    attr_np = attribution.cpu().numpy()
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
    flat_attr = attribution.view(-1).cpu()  # Move to CPU for processing
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

    # Ensure all tensors are on the same device for indexing operations
    importance = attribution.abs().cpu()  # Move to CPU for indexing
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

# =========================
# Identify a correctly classified test sample
# =========================
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

# =========================
# COMPUTE TSA
# =========================
# The function expects input WITHOUT batch dimension [timesteps, features]
# but will add it internally when calling model.forward
attribution_full = compute_temporal_spike_attribution_analytical(
    model,
    sample_spike_sequence,  # [timesteps, features]
    beta=beta,
    device=device
)

# =========================
# SLICE ATTRIBUTION FOR THE PREDICTED CLASS
# =========================
attribution = attribution_full[:, :, pred_label]   # (T, input_dim)

print("\nTop attributed spikes:")
print(summarise_top_spikes(attribution, feature_cols, top_k=10))

plot_attribution_heatmap(attribution, feature_cols)

# =========================
# VERIFY USING CLASS-SPECIFIC ATTRIBUTION
# =========================
verification = verify_attribution_by_spike_editing(
    model,
    sample_spike_sequence,
    attribution,     # <-- now correct shape
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
