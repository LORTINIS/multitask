"""
Multitask Time Series SNN Training: Wine Classification + Ethanol Concentration Regression
Training script using DIRECT TIME SERIES DATA (not aggregated metrics)

This script trains a single SNN with ONE shared layer followed by separate hidden layers:
1. Wine quality classification (HQ, LQ, AQ) - with dedicated hidden layer
2. Ethanol concentration regression (1-20%) - with dedicated hidden layer

Key differences from aggregated approach:
- Uses full temporal sequences (1000 timesteps) instead of statistical features
- Direct spike encoding from time series (not latency encoding on aggregates)
- Only MQ sensor data (excludes temperature and humidity)
- Preserves temporal dynamics throughout training

Architecture: Shared (28) → Classification Hidden (8) → 3 outputs
                          → Regression Hidden (14) → 1 output
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import datetime

# Import architecture utilities
from architectures import (
    get_available_architectures,
    load_architecture,
    print_available_architectures
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Time series configuration
SEQUENCE_LENGTH = 1000
DOWNSAMPLE_FACTOR = 2
STABILIZATION_PERIOD = 500
ENCODING_TYPE = 'direct'  # Options: 'direct', 'delta', 'rate'

# Only MQ sensors (excluding temperature and humidity)
SENSOR_COLS = [
    'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
    'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
]

# Model hyperparameters
SHARED_HIDDEN = 28
CLASS_HIDDEN = 8
REG_HIDDEN = 14
NUM_CLASSES = 3
BETA = 0.9

# Training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 16
REG_WEIGHT = 0.5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_wine_timeseries_dataset(base_path):
    """
    Load wine dataset preserving temporal structure.
    Returns DataFrame with time series for each file.
    """
    all_data = []
    
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    wine_categories = {
        'HQ': 'HQ',  # High Quality
        'LQ': 'LQ',  # Low Quality
        'AQ': 'AQ'   # Average Quality
    }
    
    print(f"Loading wine data from: {base_path}")
    
    for category, label in wine_categories.items():
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            continue
            
        print(f"Processing category: {category}...")
        
        files = os.listdir(category_path)
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(category_path, file_name)
                
                try:
                    df_file = pd.read_csv(file_path, sep=r'\s+', header=None, 
                                        names=ts_columns)
                    df_file['Filename'] = file_name
                    df_file['Time_Point'] = range(len(df_file))
                    df_file['Category'] = category
                    df_file['Label'] = label
                    
                    # Extract repetition
                    rep_start_index = file_name.rfind('R')
                    df_file['Repetition'] = (file_name[rep_start_index:rep_start_index + 3] 
                                            if rep_start_index != -1 else 'Unknown')
                    
                    all_data.append(df_file)
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue
    
    if not all_data:
        return pd.DataFrame()
    
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal wine rows loaded: {len(final_df)}")
    print(f"Unique wine files: {final_df['Filename'].nunique()}")
    return final_df


def load_ethanol_timeseries_dataset(base_path):
    """
    Load ethanol dataset preserving temporal structure.
    Returns DataFrame with time series for each file.
    """
    all_data = []
    
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    ethanol_concentration_map = {
        'C1': 1.0,
        'C2': 2.5,
        'C3': 5.0,
        'C4': 10.0,
        'C5': 15.0,
        'C6': 20.0
    }
    
    print(f"Loading ethanol data from: {base_path}")
    
    ethanol_path = os.path.join(base_path, 'Ethanol')
    
    if not os.path.exists(ethanol_path):
        print(f"Warning: Ethanol folder not found at {ethanol_path}")
        return pd.DataFrame()
    
    print(f"Processing folder: Ethanol...")
    
    files = os.listdir(ethanol_path)
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(ethanol_path, file_name)
            
            try:
                df_file = pd.read_csv(file_path, sep=r'\s+', header=None, 
                                    names=ts_columns)
                df_file['Filename'] = file_name
                df_file['Time_Point'] = range(len(df_file))
                
                # Extract concentration
                conc_code = file_name[3:5]
                df_file['Concentration_Value'] = ethanol_concentration_map.get(conc_code, np.nan)
                df_file['Concentration_Label'] = f"{ethanol_concentration_map.get(conc_code, 'Unknown')}%"
                
                # Repetition
                rep_start_index = file_name.rfind('R')
                df_file['Repetition'] = (file_name[rep_start_index:rep_start_index + 3] 
                                        if rep_start_index != -1 else 'Unknown')
                
                all_data.append(df_file)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal ethanol rows loaded: {len(final_df)}")
    print(f"Unique ethanol files: {final_df['Filename'].nunique()}")
    return final_df


# ============================================================================
# TIME SERIES PREPROCESSING
# ============================================================================

def remove_stabilization_period(df, period=500):
    """Remove initial stabilization period from each file."""
    processed_files = []
    
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename].copy()
        
        if len(file_data) > period:
            file_data = file_data.iloc[period:].reset_index(drop=True)
            processed_files.append(file_data)
    
    return pd.concat(processed_files, ignore_index=True) if processed_files else pd.DataFrame()


def prepare_time_series_sequences(df, feature_cols, sequence_length, downsample_factor):
    """
    Prepare time series sequences from DataFrame.
    
    Returns:
        sequences: List of numpy arrays [timesteps, features]
        targets: List of target values (labels or concentrations)
        metadata: List of metadata dicts
    """
    sequences = []
    targets = []
    metadata = []
    
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename]
        
        # Extract time series
        sequence = file_data[feature_cols].values
        
        # Downsample
        if downsample_factor > 1:
            sequence = sequence[::downsample_factor]
        
        # Fix length
        if len(sequence) < sequence_length:
            padding = np.repeat(sequence[-1:], sequence_length - len(sequence), axis=0)
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]
        
        sequences.append(sequence)
        
        # Determine target based on dataset type
        if 'Label' in file_data.columns:  # Wine dataset
            targets.append(file_data['Label'].iloc[0])
        elif 'Concentration_Value' in file_data.columns:  # Ethanol dataset
            targets.append(file_data['Concentration_Value'].iloc[0])
        
        metadata.append({
            'Filename': filename,
            'Repetition': file_data['Repetition'].iloc[0]
        })
    
    return sequences, targets, metadata


def clean_and_normalize_sequences(sequences):
    """Clean missing values and normalize sequences to [0, 1]."""
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
    
    # Normalize
    all_data = np.vstack(sequences)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_data)
    
    normalized_sequences = [scaler.transform(seq) for seq in sequences]
    
    return normalized_sequences, scaler


# ============================================================================
# SPIKE ENCODING FUNCTIONS
# ============================================================================

def encode_time_series_direct(sequences):
    """
    Direct threshold-based spike encoding.
    Input: List of [timesteps, features] arrays
    Output: [timesteps, samples, features] tensor
    """
    sequences_tensor = torch.FloatTensor(np.array(sequences))
    
    # Threshold-based: values > 0.5 become spikes
    spike_data = (sequences_tensor > 0.5).float()
    
    # Reshape to [timesteps, samples, features]
    spike_data = spike_data.permute(1, 0, 2)
    
    return spike_data


def encode_time_series_delta(sequences, threshold=0.05):
    """
    Delta encoding - spikes represent significant temporal changes.
    """
    sequences_tensor = torch.FloatTensor(np.array(sequences))
    
    # Compute differences
    deltas = sequences_tensor[:, 1:, :] - sequences_tensor[:, :-1, :]
    
    # Normalize deltas
    min_val, max_val = deltas.min(), deltas.max()
    deltas = (deltas - min_val) / (max_val - min_val + 1e-8)
    
    # Apply threshold
    spike_data = (torch.abs(deltas) > threshold).float()
    
    # Permute to [timesteps, samples, features]
    spike_data = spike_data.permute(1, 0, 2)
    
    return spike_data


def encode_time_series_rate(sequences, num_steps=100):
    """
    Rate encoding - spike frequency represents value.
    """
    sequences_tensor = torch.FloatTensor(np.array(sequences))
    
    # Normalize if needed
    min_val, max_val = sequences_tensor.min(), sequences_tensor.max()
    if max_val > 1.0 or min_val < 0.0:
        sequences_tensor = (sequences_tensor - min_val) / (max_val - min_val + 1e-8)
    
    # Average over time
    avg_input = sequences_tensor.mean(dim=1)
    
    # Generate rate-encoded spikes
    rand = torch.rand((num_steps, *avg_input.shape))
    spike_data = (rand < avg_input.unsqueeze(0)).float()
    
    return spike_data


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_multitask_epoch(model, wine_loader, ethanol_loader, device,
                          classification_criterion, regression_criterion,
                          optimizer, reg_weight=0.5):
    """Train for one epoch using both tasks."""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    wine_iter = iter(wine_loader)
    ethanol_iter = iter(ethanol_loader)
    
    wine_exhausted = False
    ethanol_exhausted = False
    
    while not (wine_exhausted and ethanol_exhausted):
        batch_loss = 0
        batch_class_loss = 0
        batch_reg_loss = 0
        tasks_processed = 0
        
        # Process wine batch (classification)
        if not wine_exhausted:
            try:
                wine_spikes, wine_labels = next(wine_iter)
                wine_spikes = wine_spikes.to(device)
                wine_labels = wine_labels.to(device)
                
                # Permute to [time_steps, batch, features]
                wine_spikes = wine_spikes.permute(1, 0, 2)
                
                # Forward pass
                output = model(wine_spikes)
                
                # Classification loss
                class_mem_rec = output['classification']['mem_rec']
                class_loss = classification_criterion(
                    class_mem_rec.sum(0), wine_labels.long()
                )
                
                batch_class_loss = class_loss.item()
                batch_loss += (1 - reg_weight) * class_loss
                tasks_processed += 1
                
            except StopIteration:
                wine_exhausted = True
        
        # Process ethanol batch (regression)
        if not ethanol_exhausted:
            try:
                ethanol_spikes, ethanol_conc = next(ethanol_iter)
                ethanol_spikes = ethanol_spikes.to(device)
                ethanol_conc = ethanol_conc.to(device)
                
                # Permute to [time_steps, batch, features]
                ethanol_spikes = ethanol_spikes.permute(1, 0, 2)
                
                # Forward pass
                output = model(ethanol_spikes)
                
                # Regression loss
                reg_mem_rec = output['regression']['mem_rec']
                predicted_conc = reg_mem_rec.sum(0)
                reg_loss = regression_criterion(predicted_conc.squeeze(), ethanol_conc)
                
                batch_reg_loss = reg_loss.item()
                batch_loss += reg_weight * reg_loss
                tasks_processed += 1
                
            except StopIteration:
                ethanol_exhausted = True
        
        # Backpropagation
        if tasks_processed > 0:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            total_class_loss += batch_class_loss
            total_reg_loss += batch_reg_loss
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_class_loss = total_class_loss / num_batches if num_batches > 0 else 0
    avg_reg_loss = total_reg_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_class_loss, avg_reg_loss


def evaluate_classification(model, data_loader, device):
    """Evaluate classification performance."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spikes, labels in data_loader:
            spikes = spikes.to(device)
            labels = labels.to(device)
            
            spikes = spikes.permute(1, 0, 2)
            
            output = model(spikes)
            mem_rec = output['classification']['mem_rec']
            
            _, predicted = torch.max(mem_rec.sum(0), 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def evaluate_regression(model, data_loader, device):
    """Evaluate regression performance."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for spikes, targets in data_loader:
            spikes = spikes.to(device)
            targets = targets.to(device)
            
            spikes = spikes.permute(1, 0, 2)
            
            output = model(spikes)
            mem_rec = output['regression']['mem_rec']
            
            predicted = mem_rec.sum(0).squeeze()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['total_loss'], label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['class_loss'], label='Classification Loss',
                 color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Classification Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['reg_loss'], label='Regression Loss',
                 color='teal', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Regression Loss', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(results, class_names):
    """Plot classification metrics and confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1']
    }
    
    axes[0].bar(metrics.keys(), metrics.values(), 
               color=['blue', 'green', 'orange', 'red'])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classification Metrics', fontsize=14, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    for i, (metric, value) in enumerate(metrics.items()):
        axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=10)
    
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_regression_results(results, scaler_y=None):
    """Plot regression metrics and predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Denormalize if scaler provided
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(
            results['predictions'].reshape(-1, 1)).flatten()
        targets = scaler_y.inverse_transform(
            results['targets'].reshape(-1, 1)).flatten()
        
        # Recalculate metrics in original scale
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
    else:
        predictions = results['predictions']
        targets = results['targets']
        rmse = results['rmse']
        mae = results['mae']
        r2 = results['r2']
    
    metrics = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    colors = ['red', 'orange', 'green']
    bars = axes[0].bar(metrics.keys(), metrics.values(), color=colors)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Regression Metrics', fontsize=14, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    for bar, (metric, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', fontsize=10)
    
    axes[1].scatter(targets, predictions, alpha=0.6, s=50)
    
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Concentration (%)', fontsize=12)
    axes[1].set_ylabel('Predicted Concentration (%)', fontsize=12)
    axes[1].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(selected_arch_name=None):
    """Main training pipeline."""
    print("="*80)
    print("MULTITASK TIME SERIES SNN")
    print("Wine Classification + Ethanol Regression")
    print("="*80)
    
    # Architecture selection
    if selected_arch_name is None:
        print_available_architectures()
        available_archs = get_available_architectures()
        
        if not available_archs:
            print("\nERROR: No architectures found!")
            sys.exit(1)
        
        arch_list = list(available_archs.keys())
        print("\nSelect an architecture:")
        for i, arch_name in enumerate(arch_list, 1):
            print(f"  {i}. {arch_name}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(arch_list)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(arch_list):
                    selected_arch_name = arch_list[choice_idx]
                    break
            except (ValueError, KeyboardInterrupt):
                print("\nTraining cancelled.")
                sys.exit(0)
    
    print(f"\n✓ Selected architecture: {selected_arch_name}")
    
    # Load architecture
    arch_module = load_architecture(selected_arch_name)
    MultitaskSNN = arch_module.MultitaskSNN
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "model")
    os.makedirs(models_dir, exist_ok=True)
    data_path = os.path.join(base_dir, "data", "wine")
    
    print(f"\nUsing device: {device}")
    print(f"Data path: {data_path}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING TIME SERIES DATA")
    print("="*80)
    
    print("\nLoading wine dataset...")
    wine_df = load_wine_timeseries_dataset(data_path)
    
    if wine_df.empty:
        print(f"\nERROR: No wine data found!")
        sys.exit(1)
    
    print("\nLoading ethanol dataset...")
    ethanol_df = load_ethanol_timeseries_dataset(data_path)
    
    if ethanol_df.empty:
        print(f"\nERROR: No ethanol data found!")
        sys.exit(1)
    
    # Remove stabilization period
    print(f"\nRemoving first {STABILIZATION_PERIOD} time points...")
    wine_df = remove_stabilization_period(wine_df, STABILIZATION_PERIOD)
    ethanol_df = remove_stabilization_period(ethanol_df, STABILIZATION_PERIOD)
    
    # ========================================================================
    # PREPARE TIME SERIES
    # ========================================================================
    print("\n" + "="*80)
    print("PREPARING TIME SERIES SEQUENCES")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  Features: {len(SENSOR_COLS)} (MQ sensors only)")
    print(f"  Encoding: {ENCODING_TYPE}")
    
    # Wine sequences
    print("\nPreparing wine sequences...")
    wine_sequences, wine_labels, wine_metadata = prepare_time_series_sequences(
        wine_df, SENSOR_COLS, SEQUENCE_LENGTH, DOWNSAMPLE_FACTOR
    )
    
    # Ethanol sequences
    print("Preparing ethanol sequences...")
    ethanol_sequences, ethanol_conc, ethanol_metadata = prepare_time_series_sequences(
        ethanol_df, SENSOR_COLS, SEQUENCE_LENGTH, DOWNSAMPLE_FACTOR
    )
    
    print(f"\nWine sequences: {len(wine_sequences)}")
    print(f"Ethanol sequences: {len(ethanol_sequences)}")
    
    # Clean and normalize
    print("\nNormalizing wine sequences...")
    wine_sequences_norm, wine_scaler = clean_and_normalize_sequences(wine_sequences)
    
    print("Normalizing ethanol sequences...")
    ethanol_sequences_norm, ethanol_scaler = clean_and_normalize_sequences(ethanol_sequences)
    
    # Encode labels
    wine_label_encoder = LabelEncoder()
    wine_labels_encoded = wine_label_encoder.fit_transform(wine_labels)
    
    # Normalize ethanol concentrations
    ethanol_scaler_y = MinMaxScaler(feature_range=(0, 1))
    ethanol_conc_norm = ethanol_scaler_y.fit_transform(
        np.array(ethanol_conc).reshape(-1, 1)
    ).flatten()
    
    print(f"\nWine classes: {wine_label_encoder.classes_}")
    print(f"Ethanol concentration range: {min(ethanol_conc):.1f}% - {max(ethanol_conc):.1f}%")
    
    # ========================================================================
    # TRAIN-TEST SPLIT
    # ========================================================================
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)
    
    # Wine split
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine_sequences_norm, wine_labels_encoded,
        test_size=0.2, random_state=42, stratify=wine_labels_encoded
    )
    
    # Ethanol split
    X_train_ethanol, X_test_ethanol, y_train_ethanol, y_test_ethanol = train_test_split(
        ethanol_sequences_norm, ethanol_conc_norm,
        test_size=0.2, random_state=42
    )
    
    print(f"\nWine - Train: {len(X_train_wine)}, Test: {len(X_test_wine)}")
    print(f"Ethanol - Train: {len(X_train_ethanol)}, Test: {len(X_test_ethanol)}")
    
    # ========================================================================
    # SPIKE ENCODING
    # ========================================================================
    print("\n" + "="*80)
    print("SPIKE ENCODING")
    print("="*80)
    
    print(f"\nEncoding type: {ENCODING_TYPE}")
    
    if ENCODING_TYPE == 'direct':
        print("\nEncoding wine training data...")
        wine_train_spikes = encode_time_series_direct(X_train_wine)
        print("Encoding wine test data...")
        wine_test_spikes = encode_time_series_direct(X_test_wine)
        
        print("\nEncoding ethanol training data...")
        ethanol_train_spikes = encode_time_series_direct(X_train_ethanol)
        print("Encoding ethanol test data...")
        ethanol_test_spikes = encode_time_series_direct(X_test_ethanol)
        
    elif ENCODING_TYPE == 'delta':
        print("\nEncoding wine training data...")
        wine_train_spikes = encode_time_series_delta(X_train_wine)
        print("Encoding wine test data...")
        wine_test_spikes = encode_time_series_delta(X_test_wine)
        
        print("\nEncoding ethanol training data...")
        ethanol_train_spikes = encode_time_series_delta(X_train_ethanol)
        print("Encoding ethanol test data...")
        ethanol_test_spikes = encode_time_series_delta(X_test_ethanol)
        
    else:  # rate encoding
        print("\nEncoding wine training data...")
        wine_train_spikes = encode_time_series_rate(X_train_wine, num_steps=100)
        print("Encoding wine test data...")
        wine_test_spikes = encode_time_series_rate(X_test_wine, num_steps=100)
        
        print("\nEncoding ethanol training data...")
        ethanol_train_spikes = encode_time_series_rate(X_train_ethanol, num_steps=100)
        print("Encoding ethanol test data...")
        ethanol_test_spikes = encode_time_series_rate(X_test_ethanol, num_steps=100)
    
    print(f"\nWine spikes shape: {wine_train_spikes.shape}")
    print(f"Ethanol spikes shape: {ethanol_train_spikes.shape}")
    
    # Calculate spike statistics
    wine_spike_rate = wine_train_spikes.sum().item() / wine_train_spikes.numel()
    ethanol_spike_rate = ethanol_train_spikes.sum().item() / ethanol_train_spikes.numel()
    
    print(f"\nWine spike rate: {wine_spike_rate:.4f}")
    print(f"Ethanol spike rate: {ethanol_spike_rate:.4f}")
    
    # ========================================================================
    # CREATE DATALOADERS
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)
    
    # Permute to [batch, time, features] for DataLoader
    wine_train_spikes = wine_train_spikes.permute(1, 0, 2)
    wine_test_spikes = wine_test_spikes.permute(1, 0, 2)
    ethanol_train_spikes = ethanol_train_spikes.permute(1, 0, 2)
    ethanol_test_spikes = ethanol_test_spikes.permute(1, 0, 2)
    
    wine_train_loader, wine_test_loader = create_dataloaders(
        wine_train_spikes, torch.LongTensor(y_train_wine),
        wine_test_spikes, torch.LongTensor(y_test_wine),
        batch_size=BATCH_SIZE
    )
    
    ethanol_train_loader, ethanol_test_loader = create_dataloaders(
        ethanol_train_spikes, torch.FloatTensor(y_train_ethanol),
        ethanol_test_spikes, torch.FloatTensor(y_test_ethanol),
        batch_size=BATCH_SIZE
    )
    
    print(f"\nWine train batches: {len(wine_train_loader)}")
    print(f"Ethanol train batches: {len(ethanol_train_loader)}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING MULTITASK SNN MODEL")
    print("="*80)
    
    input_size = len(SENSOR_COLS)
    
    model = MultitaskSNN(
        input_size=input_size,
        shared_hidden=SHARED_HIDDEN,
        classification_hidden=CLASS_HIDDEN,
        regression_hidden=REG_HIDDEN,
        num_classes=NUM_CLASSES,
        beta=BETA
    ).to(device)
    
    print(model.get_architecture_summary())
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING MULTITASK SNN")
    print("="*80)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Regression weight: {REG_WEIGHT}")
    
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'total_loss': [],
        'class_loss': [],
        'reg_loss': []
    }
    
    print("\nStarting training...")
    print("-" * 80)
    
    for epoch in range(NUM_EPOCHS):
        avg_loss, avg_class_loss, avg_reg_loss = train_multitask_epoch(
            model, wine_train_loader, ethanol_train_loader, device,
            classification_criterion, regression_criterion,
            optimizer, reg_weight=REG_WEIGHT
        )
        
        history['total_loss'].append(avg_loss)
        history['class_loss'].append(avg_class_loss)
        history['reg_loss'].append(avg_reg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
                  f"Total: {avg_loss:.4f} | "
                  f"Class: {avg_class_loss:.4f} | "
                  f"Reg: {avg_reg_loss:.4f}")
    
    print("\n✓ Training completed!")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Classification
    print("\n--- WINE CLASSIFICATION RESULTS ---")
    class_results = evaluate_classification(model, wine_test_loader, device)
    print(f"Accuracy:  {class_results['accuracy']:.4f}")
    print(f"Precision: {class_results['precision']:.4f}")
    print(f"Recall:    {class_results['recall']:.4f}")
    print(f"F1-Score:  {class_results['f1']:.4f}")
    
    # Regression
    print("\n--- ETHANOL CONCENTRATION REGRESSION RESULTS ---")
    reg_results = evaluate_regression(model, ethanol_test_loader, device)
    
    # Denormalize for display
    reg_results_denorm = reg_results.copy()
    reg_results_denorm['predictions'] = ethanol_scaler_y.inverse_transform(
        reg_results['predictions'].reshape(-1, 1)).flatten()
    reg_results_denorm['targets'] = ethanol_scaler_y.inverse_transform(
        reg_results['targets'].reshape(-1, 1)).flatten()
    
    rmse_denorm = np.sqrt(mean_squared_error(
        reg_results_denorm['targets'], reg_results_denorm['predictions']))
    mae_denorm = mean_absolute_error(
        reg_results_denorm['targets'], reg_results_denorm['predictions'])
    r2_denorm = r2_score(
        reg_results_denorm['targets'], reg_results_denorm['predictions'])
    
    print(f"RMSE: {rmse_denorm:.4f}%")
    print(f"MAE:  {mae_denorm:.4f}%")
    print(f"R²:   {r2_denorm:.4f}")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = (f"{selected_arch_name}_timeseries_"
                  f"seq{SEQUENCE_LENGTH}_"
                  f"enc{ENCODING_TYPE}_"
                  f"beta{BETA}_"
                  f"epochs{NUM_EPOCHS}_"
                  f"lr{LEARNING_RATE}_"
                  f"bs{BATCH_SIZE}_"
                  f"rw{REG_WEIGHT}_"
                  f"{timestamp}.pth")
    
    model_path = os.path.join(models_dir, model_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture_name': selected_arch_name,
        'architecture': {
            'input_size': input_size,
            'shared_hidden': SHARED_HIDDEN,
            'classification_hidden': CLASS_HIDDEN,
            'regression_hidden': REG_HIDDEN,
            'num_classes': NUM_CLASSES,
            'beta': BETA
        },
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'reg_weight': REG_WEIGHT,
            'sequence_length': SEQUENCE_LENGTH,
            'downsample_factor': DOWNSAMPLE_FACTOR,
            'encoding_type': ENCODING_TYPE
        },
        'results': {
            'classification': class_results,
            'regression': {
                'mse': reg_results['mse'],
                'rmse': rmse_denorm,
                'mae': mae_denorm,
                'r2': r2_denorm,
                'predictions': reg_results_denorm['predictions'],
                'targets': reg_results_denorm['targets']
            }
        },
        'history': history,
        'scalers': {
            'wine_scaler': wine_scaler,
            'ethanol_scaler': ethanol_scaler,
            'ethanol_scaler_y': ethanol_scaler_y
        },
        'label_encoder': wine_label_encoder,
        'sensor_cols': SENSOR_COLS
    }, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Training history
    plot_training_history(history)
    
    # Classification results
    class_names = wine_label_encoder.classes_
    plot_classification_results(class_results, class_names)
    
    # Regression results
    plot_regression_results(reg_results, ethanol_scaler_y)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("MULTITASK TIME SERIES SNN - FINAL SUMMARY")
    print("="*80)
    
    print(f"""
✓ TRAINING COMPLETED SUCCESSFULLY!

APPROACH: TIME SERIES DIRECT PROCESSING
========================================
- Full temporal sequences ({SEQUENCE_LENGTH} timesteps)
- Direct spike encoding (not aggregated statistics)
- Only MQ sensor data (6 features, no temperature/humidity)
- Preserves temporal dynamics throughout training

ARCHITECTURE:
=============
- Input: {input_size} features per timestep
- Shared hidden: {SHARED_HIDDEN} neurons
- Classification hidden: {CLASS_HIDDEN} neurons
- Regression hidden: {REG_HIDDEN} neurons
- Total timesteps: {SEQUENCE_LENGTH}
- Encoding: {ENCODING_TYPE}

CLASSIFICATION RESULTS (Wine Quality):
======================================
- Accuracy:  {class_results['accuracy']:.4f}
- Precision: {class_results['precision']:.4f}
- Recall:    {class_results['recall']:.4f}
- F1-Score:  {class_results['f1']:.4f}

REGRESSION RESULTS (Ethanol Concentration):
==========================================
- RMSE: {rmse_denorm:.4f}%
- MAE:  {mae_denorm:.4f}%
- R²:   {r2_denorm:.4f}

DATASET:
========
- Wine samples: {len(wine_sequences)}
- Ethanol samples: {len(ethanol_sequences)}
- Train split: 80%
- Test split: 20%
- Features: {len(SENSOR_COLS)} MQ sensors
- Sequence length: {SEQUENCE_LENGTH} timesteps
- Downsample factor: {DOWNSAMPLE_FACTOR}

TRAINING:
=========
- Epochs: {NUM_EPOCHS}
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}
- Regression weight: {REG_WEIGHT}
- Total parameters: {total_params:,}

KEY ADVANTAGES:
===============
✓ Temporal dynamics preserved (no aggregation)
✓ Biological sensing patterns maintained
✓ SNN temporal processing fully utilized
✓ Multitask learning with shared representations
✓ Direct spike encoding from time series
✓ Only relevant sensor features (no environmental)

Model saved at: {model_path}
""")
    
    print("\n" + "="*80)
    print("ALL DONE! ✅")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(selected_arch_name=sys.argv[1])
    else:
        main()