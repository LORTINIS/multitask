"""
Multitask SNN Training: Wine Classification + Ethanol Concentration Regression.
Training script for TIME SERIES DATA with SINGLE SHARED LAYER architectures.

This script trains a single SNN with ONE shared layer followed by separate hidden layers:
1. Wine quality classification (HQ, LQ, AQ) - with dedicated hidden layer
2. Ethanol concentration regression (1-20%) - with dedicated hidden layer

Architecture: Shared (28) → Classification Hidden (8) → 3 outputs
                          → Regression Hidden (14) → 1 output
                          
UPDATED: Now uses full time series data instead of aggregated features!
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
import seaborn as sns
from datetime import datetime

# Import utilities - UPDATED for time series
from utils.data_loader import load_wine_dataset, load_ethanol_dataset
from utils.preprocessing_time_series import (
    preprocess_time_series_features,
    create_train_test_split_sequences
)
from utils.spike_encoding_time_series import encode_time_series_datasets

# Import architecture utilities
from architectures import (
    get_available_architectures,
    load_architecture,
    print_available_architectures
)


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_multitask_epoch(model, wine_loader, ethanol_loader, device,
                          classification_criterion, regression_criterion,
                          optimizer, reg_weight=0.5):
    """
    Train for one epoch using both tasks.
    
    Args:
        model: MultitaskSNN model
        wine_loader: DataLoader for wine classification
        ethanol_loader: DataLoader for ethanol regression
        device: torch device
        classification_criterion: Loss function for classification
        regression_criterion: Loss function for regression
        optimizer: Optimizer
        reg_weight: Weight for regression loss (classification weight = 1 - reg_weight)
    
    Returns:
        Average loss, classification loss, regression loss
    """
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    # Create iterators
    wine_iter = iter(wine_loader)
    ethanol_iter = iter(ethanol_loader)
    
    # Train on batches from both datasets
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
                
                # Permute to [time_steps, batch, features] for model forward pass
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
                
                # Permute to [time_steps, batch, features] for model forward pass
                ethanol_spikes = ethanol_spikes.permute(1, 0, 2)
                
                # Forward pass
                output = model(ethanol_spikes)
                
                # Regression loss
                reg_mem_rec = output['regression']['mem_rec']
                predicted_conc = reg_mem_rec.sum(0)  # Sum over time
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
            
            # Permute to [time_steps, batch, features]
            spikes = spikes.permute(1, 0, 2)
            
            output = model(spikes)
            mem_rec = output['classification']['mem_rec']
            
            # Predict based on accumulated membrane potential
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
            
            # Permute to [time_steps, batch, features]
            spikes = spikes.permute(1, 0, 2)
            
            output = model(spikes)
            mem_rec = output['regression']['mem_rec']
            
            # Predicted concentration is sum of membrane potential over time
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


def plot_training_history(history):
    """Plot training history for both tasks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total loss
    axes[0].plot(history['total_loss'], label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Classification loss
    axes[1].plot(history['class_loss'], label='Classification Loss',
                 color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Classification Loss (Wine Quality)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Regression loss
    axes[2].plot(history['reg_loss'], label='Regression Loss',
                 color='teal', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Regression Loss (Ethanol Concentration)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(results, class_names=['HQ', 'LQ', 'AQ']):
    """Plot classification performance metrics and confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics bar plot
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1']
    }
    
    axes[0].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classification Metrics (Wine Quality)', fontsize=14, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (metric, value) in enumerate(metrics.items()):
        axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=10)
    
    # Confusion matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Wine Quality)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_regression_results(reg_results):
    """Plot regression performance metrics and predictions vs actual."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics bar plot
    metrics = {
        'RMSE': reg_results['rmse'],
        'MAE': reg_results['mae'],
        'R²': reg_results['r2']
    }
    
    colors = ['red', 'orange', 'green']
    bars = axes[0].bar(metrics.keys(), metrics.values(), color=colors)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Regression Metrics (Ethanol Concentration)', fontsize=14, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, (metric, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', fontsize=10)
    
    # Predictions vs Actual
    predictions = reg_results['predictions']
    targets = reg_results['targets']
    
    axes[1].scatter(targets, predictions, alpha=0.6, s=50)
    
    # Plot diagonal line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Concentration (normalized)', fontsize=12)
    axes[1].set_ylabel('Predicted Concentration (normalized)', fontsize=12)
    axes[1].set_title('Predictions vs Actual (Ethanol)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_shared_layer_activity(model, wine_sample, ethanol_sample, device):
    """Visualize shared layer activity for both tasks."""
    model.eval()
    
    with torch.no_grad():
        # Wine sample
        wine_output = model(wine_sample.to(device))
        
        # Ethanol sample
        ethanol_output = model(ethanol_sample.to(device))
    
    # Check if architecture has one or two shared layers
    shared_data = wine_output['shared']
    
    if 'spk_rec1' in shared_data and 'spk_rec2' in shared_data:
        # Two shared layers architecture
        wine_shared1 = shared_data['spk_rec1'].cpu().numpy()
        wine_shared2 = shared_data['spk_rec2'].cpu().numpy()
        ethanol_shared1 = ethanol_output['shared']['spk_rec1'].cpu().numpy()
        ethanol_shared2 = ethanol_output['shared']['spk_rec2'].cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Wine - Shared Layer 1
        axes[0, 0].imshow(wine_shared1[:, 0, :].T, aspect='auto', cmap='binary')
        axes[0, 0].set_xlabel('Time Step', fontsize=10)
        axes[0, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 0].set_title('Wine Sample: Shared Layer 1', fontsize=12, fontweight='bold')
        
        # Wine - Shared Layer 2
        axes[0, 1].imshow(wine_shared2[:, 0, :].T, aspect='auto', cmap='binary')
        axes[0, 1].set_xlabel('Time Step', fontsize=10)
        axes[0, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 1].set_title('Wine Sample: Shared Layer 2', fontsize=12, fontweight='bold')
        
        # Ethanol - Shared Layer 1
        axes[1, 0].imshow(ethanol_shared1[:, 0, :].T, aspect='auto', cmap='binary')
        axes[1, 0].set_xlabel('Time Step', fontsize=10)
        axes[1, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 0].set_title('Ethanol Sample: Shared Layer 1', fontsize=12, fontweight='bold')
        
        # Ethanol - Shared Layer 2
        axes[1, 1].imshow(ethanol_shared2[:, 0, :].T, aspect='auto', cmap='binary')
        axes[1, 1].set_xlabel('Time Step', fontsize=10)
        axes[1, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 1].set_title('Ethanol Sample: Shared Layer 2', fontsize=12, fontweight='bold')
    
    elif 'spk_rec' in shared_data:
        # Single shared layer architecture
        wine_shared = shared_data['spk_rec'].cpu().numpy()
        ethanol_shared = ethanol_output['shared']['spk_rec'].cpu().numpy()
        
        # Also get task-specific hidden layers
        wine_class_hidden = wine_output['classification']['spk_rec'].cpu().numpy()
        wine_reg_hidden = wine_output['regression']['spk_rec'].cpu().numpy()
        ethanol_class_hidden = ethanol_output['classification']['spk_rec'].cpu().numpy()
        ethanol_reg_hidden = ethanol_output['regression']['spk_rec'].cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Wine - Shared Layer
        axes[0, 0].imshow(wine_shared[:, 0, :].T, aspect='auto', cmap='binary')
        axes[0, 0].set_xlabel('Time Step', fontsize=10)
        axes[0, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 0].set_title('Wine: Shared Layer', fontsize=12, fontweight='bold')
        
        # Wine - Classification Hidden
        axes[0, 1].imshow(wine_class_hidden[:, 0, :].T, aspect='auto', cmap='Reds')
        axes[0, 1].set_xlabel('Time Step', fontsize=10)
        axes[0, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 1].set_title('Wine: Classification Hidden', fontsize=12, fontweight='bold')
        
        # Wine - Regression Hidden
        axes[0, 2].imshow(wine_reg_hidden[:, 0, :].T, aspect='auto', cmap='Blues')
        axes[0, 2].set_xlabel('Time Step', fontsize=10)
        axes[0, 2].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 2].set_title('Wine: Regression Hidden', fontsize=12, fontweight='bold')
        
        # Ethanol - Shared Layer
        axes[1, 0].imshow(ethanol_shared[:, 0, :].T, aspect='auto', cmap='binary')
        axes[1, 0].set_xlabel('Time Step', fontsize=10)
        axes[1, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 0].set_title('Ethanol: Shared Layer', fontsize=12, fontweight='bold')
        
        # Ethanol - Classification Hidden
        axes[1, 1].imshow(ethanol_class_hidden[:, 0, :].T, aspect='auto', cmap='Reds')
        axes[1, 1].set_xlabel('Time Step', fontsize=10)
        axes[1, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 1].set_title('Ethanol: Classification Hidden', fontsize=12, fontweight='bold')
        
        # Ethanol - Regression Hidden
        axes[1, 2].imshow(ethanol_reg_hidden[:, 0, :].T, aspect='auto', cmap='Blues')
        axes[1, 2].set_xlabel('Time Step', fontsize=10)
        axes[1, 2].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 2].set_title('Ethanol: Regression Hidden', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def main(selected_arch_name=None):
    """Main training and evaluation pipeline."""
    print("="*80)
    print("MULTITASK SNN: WINE CLASSIFICATION + ETHANOL REGRESSION")
    print("TIME SERIES DATA WITH SINGLE SHARED LAYER ARCHITECTURE")
    print("="*80)
    
    # ========================================================================
    # ARCHITECTURE SELECTION
    # ========================================================================
    if selected_arch_name is None:
        # Interactive mode - let user select
        print_available_architectures()
        
        available_archs = get_available_architectures()
        
        if not available_archs:
            print("\nERROR: No architectures found in the architectures directory!")
            print("Please add architecture files to the 'architectures' folder.")
            sys.exit(1)
        
        # Let user select architecture
        arch_list = list(available_archs.keys())
        print("\nSelect an architecture to train:")
        for i, arch_name in enumerate(arch_list, 1):
            print(f"  {i}. {arch_name}")
        
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(arch_list)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(arch_list):
                    selected_arch_name = arch_list[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(arch_list)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n\nTraining cancelled by user.")
                sys.exit(0)
    
    print(f"\n✓ Selected architecture: {selected_arch_name}")
    
    # Load the selected architecture module
    arch_module = load_architecture(selected_arch_name)
    MultitaskSNN = arch_module.MultitaskSNN
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    # Paths (relative to multitask_net directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to multitask folder
    wine_data_path = os.path.join(base_dir, "data", "wine")
    ethanol_data_path = os.path.join(base_dir, "data", "wine")  # Ethanol is inside wine folder
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nData paths:")
    print(f"  Wine: {wine_data_path}")
    print(f"  Ethanol: {ethanol_data_path}")
    
    # ========================================================================
    # TIME SERIES HYPERPARAMETERS - NEW!
    # ========================================================================
    SEQUENCE_LENGTH = 500       # Fixed length for sequences (or None for variable)
    DOWNSAMPLE_FACTOR = 2       # Take every Nth point (1=no downsampling)
    ENCODING_TYPE = 'direct'    # Options: 'direct', 'rate', 'latency', 'delta'
    
    # Model hyperparameters
    SHARED_HIDDEN = 28          # Single shared layer
    CLASS_HIDDEN = 8            # Classification branch hidden layer
    REG_HIDDEN = 14             # Regression branch hidden layer
    NUM_CLASSES = 3
    BETA = 0.9
    
    # Training hyperparameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    REG_WEIGHT = 0.5           # Weight for regression loss (0.5 = equal importance)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print(f"\nTime Series Configuration:")
    print(f"  Sequence length: {SEQUENCE_LENGTH if SEQUENCE_LENGTH else 'Variable'}")
    print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  Encoding type: {ENCODING_TYPE}")
    
    # ========================================================================
    # LOAD AND PREPROCESS DATA
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load wine data (classification)
    print("\nLoading wine dataset...")
    wine_df = load_wine_dataset(wine_data_path)
    
    if wine_df.empty:
        print(f"\nERROR: No wine data found at {wine_data_path}")
        print("Please check the data path and folder structure.")
        sys.exit(1)
    
    # Load ethanol data (regression)
    print("\nLoading ethanol dataset...")
    ethanol_df = load_ethanol_dataset(ethanol_data_path)
    
    if ethanol_df.empty:
        print(f"\nERROR: No ethanol data found at {ethanol_data_path}")
        print("Please check the data path and folder structure.")
        sys.exit(1)
    
    # Remove stabilization period from wine data
    print(f"\nRemoving first 500 time points from wine data...")
    wine_processed = []
    for filename in wine_df['Filename'].unique():
        file_data = wine_df[wine_df['Filename'] == filename].copy()
        if len(file_data) > 500:
            file_data = file_data.iloc[500:].reset_index(drop=True)
            wine_processed.append(file_data)
    
    wine_df = pd.concat(wine_processed, ignore_index=True) if wine_processed else pd.DataFrame()
    
    # Remove stabilization period from ethanol data
    print(f"Removing first 500 time points from ethanol data...")
    ethanol_processed = []
    for filename in ethanol_df['Filename'].unique():
        file_data = ethanol_df[ethanol_df['Filename'] == filename].copy()
        if len(file_data) > 500:
            file_data = file_data.iloc[500:].reset_index(drop=True)
            ethanol_processed.append(file_data)
    
    ethanol_df = pd.concat(ethanol_processed, ignore_index=True) if ethanol_processed else pd.DataFrame()
    
    # NEW: Preprocess time series data (preserves temporal structure)
    preprocessed_data = preprocess_time_series_features(
        wine_df, 
        ethanol_df,
        sequence_length=SEQUENCE_LENGTH,
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    # NEW: Create train-test splits for sequences
    splits = create_train_test_split_sequences(
        preprocessed_data, 
        test_size=0.2, 
        random_state=42
    )
    
    # ========================================================================
    # SPIKE ENCODING - UPDATED FOR TIME SERIES
    # ========================================================================
    print("\n" + "="*80)
    print("ENCODING SPIKE TRAINS FROM TIME SERIES")
    print("="*80)
    
    # NEW: Encode time series as spikes
    spike_encoded_data = encode_time_series_datasets(
        splits,
        encoding_type=ENCODING_TYPE,
        num_steps=None  # Only needed for 'rate' or 'latency'
    )
    
    # Extract spike trains (already in correct format!)
    X_train_wine_spikes = spike_encoded_data['wine']['spike_train']
    X_test_wine_spikes = spike_encoded_data['wine']['spike_test']
    y_train_wine = spike_encoded_data['wine']['y_train']
    y_test_wine = spike_encoded_data['wine']['y_test']
    wine_label_encoder = spike_encoded_data['wine']['label_encoder']
    
    X_train_ethanol_spikes = spike_encoded_data['ethanol']['spike_train']
    X_test_ethanol_spikes = spike_encoded_data['ethanol']['spike_test']
    y_train_ethanol = spike_encoded_data['ethanol']['y_train']
    y_test_ethanol = spike_encoded_data['ethanol']['y_test']
    ethanol_scaler_y = spike_encoded_data['ethanol']['scaler_y']
    
    print(f"\nSpike train shapes:")
    print(f"  Wine train: {X_train_wine_spikes.shape} (time_steps, samples, features)")
    print(f"  Ethanol train: {X_train_ethanol_spikes.shape} (time_steps, samples, features)")
    
    # ========================================================================
    # CREATE DATA LOADERS
    # ========================================================================
    # Permute dimensions: [time_steps, samples, features] -> [samples, time_steps, features]
    X_train_wine_spikes = X_train_wine_spikes.permute(1, 0, 2)
    X_test_wine_spikes = X_test_wine_spikes.permute(1, 0, 2)
    X_train_ethanol_spikes = X_train_ethanol_spikes.permute(1, 0, 2)
    X_test_ethanol_spikes = X_test_ethanol_spikes.permute(1, 0, 2)
    
    wine_train_loader, wine_test_loader = create_dataloaders(
        X_train_wine_spikes, y_train_wine,
        X_test_wine_spikes, y_test_wine,
        batch_size=BATCH_SIZE
    )
    
    ethanol_train_loader, ethanol_test_loader = create_dataloaders(
        X_train_ethanol_spikes, y_train_ethanol,
        X_test_ethanol_spikes, y_test_ethanol,
        batch_size=BATCH_SIZE
    )
    
    # ========================================================================
    # CREATE MODEL - UPDATED INPUT SIZE
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING MULTITASK SNN")
    print("="*80)
    
    # For time series: input_size = number of features per timestep
    input_size = splits['wine']['X_train'].shape[2]  # Last dimension = features
    
    print(f"\nModel configuration:")
    print(f"  Input size: {input_size} (features per timestep)")
    print(f"  Shared hidden: {SHARED_HIDDEN}")
    print(f"  Classification hidden: {CLASS_HIDDEN}")
    print(f"  Regression hidden: {REG_HIDDEN}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Beta: {BETA}")
    
    model = MultitaskSNN(
        input_size=input_size,
        shared_hidden=SHARED_HIDDEN,
        classification_hidden=CLASS_HIDDEN,
        regression_hidden=REG_HIDDEN,
        num_classes=NUM_CLASSES,
        beta=BETA
    ).to(device)
    
    print(model.get_architecture_summary())
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Regression weight: {REG_WEIGHT}")
    
    history = {
        'total_loss': [],
        'class_loss': [],
        'reg_loss': []
    }
    
    for epoch in range(NUM_EPOCHS):
        avg_loss, avg_class_loss, avg_reg_loss = train_multitask_epoch(
            model, wine_train_loader, ethanol_train_loader, device,
            classification_criterion, regression_criterion,
            optimizer, reg_weight=REG_WEIGHT
        )
        
        history['total_loss'].append(avg_loss)
        history['class_loss'].append(avg_class_loss)
        history['reg_loss'].append(avg_reg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Total Loss: {avg_loss:.4f} | "
                  f"Class Loss: {avg_class_loss:.4f} | "
                  f"Reg Loss: {avg_reg_loss:.4f}")
    
    print("\n✓ Training completed!")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Classification evaluation
    print("\n--- WINE CLASSIFICATION RESULTS ---")
    class_results = evaluate_classification(model, wine_test_loader, device)
    print(f"  Accuracy:  {class_results['accuracy']:.4f}")
    print(f"  Precision: {class_results['precision']:.4f}")
    print(f"  Recall:    {class_results['recall']:.4f}")
    print(f"  F1-Score:  {class_results['f1']:.4f}")
    
    # Regression evaluation
    print("\n--- ETHANOL CONCENTRATION REGRESSION RESULTS ---")
    reg_results = evaluate_regression(model, ethanol_test_loader, device)
    print(f"  RMSE: {reg_results['rmse']:.4f}")
    print(f"  MAE:  {reg_results['mae']:.4f}")
    print(f"  R²:   {reg_results['r2']:.4f}")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name with time series parameters
    model_name = (f"{selected_arch_name}_"
                  f"timeseries_"
                  f"seqlen-{SEQUENCE_LENGTH if SEQUENCE_LENGTH else 'var'}_"
                  f"downsample-{DOWNSAMPLE_FACTOR}_"
                  f"encoding-{ENCODING_TYPE}_"
                  f"beta-{BETA}_"
                  f"epochs-{NUM_EPOCHS}_"
                  f"lr-{LEARNING_RATE}_"
                  f"bs-{BATCH_SIZE}_"
                  f"regweight-{REG_WEIGHT}_"
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
            'regression': reg_results
        },
        'history': history,
        'scalers': {
            'wine_scaler': preprocessed_data['wine']['scaler_X'],
            'ethanol_scaler': preprocessed_data['ethanol']['scaler_X'],
            'ethanol_scaler_y': ethanol_scaler_y
        },
        'label_encoder': wine_label_encoder,
        'feature_names': splits['wine']['feature_names']
    }, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Training history
    print("\n1. Training history...")
    plot_training_history(history)
    
    # Classification results
    print("2. Classification results...")
    class_names = wine_label_encoder.classes_
    plot_classification_results(class_results, class_names)
    
    # Regression results
    print("3. Regression results...")
    plot_regression_results(reg_results)
    
    # Shared layer activity visualization
    print("4. Shared layer activity...")
    # Get first test sample: [batch, time, features] -> [time, 1, features]
    wine_sample = X_test_wine_spikes[0:1, :, :].permute(1, 0, 2)
    ethanol_sample = X_test_ethanol_spikes[0:1, :, :].permute(1, 0, 2)
    plot_shared_layer_activity(model, wine_sample, ethanol_sample, device)
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nArchitecture: {selected_arch_name}")
    print(f"Time Series Config:")
    print(f"  - Sequence length: {SEQUENCE_LENGTH if SEQUENCE_LENGTH else 'Variable'}")
    print(f"  - Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  - Encoding type: {ENCODING_TYPE}")
    print(f"  - Input features: {input_size}")
    print(f"\nWine Classification Performance:")
    print(f"  - Accuracy: {class_results['accuracy']:.4f}")
    print(f"  - F1-Score: {class_results['f1']:.4f}")
    print(f"\nEthanol Regression Performance:")
    print(f"  - RMSE: {reg_results['rmse']:.4f}")
    print(f"  - R²: {reg_results['r2']:.4f}")
    print(f"\nModel saved to: {model_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    # Check if architecture name was passed as command line argument
    if len(sys.argv) > 1:
        main(selected_arch_name=sys.argv[1])
    else:
        main()