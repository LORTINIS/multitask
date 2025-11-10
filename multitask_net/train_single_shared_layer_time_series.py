"""
Improved Multitask SNN Training: Wine Classification + Ethanol Concentration Regression.
Training script for TIME SERIES DATA with IMPROVED ARCHITECTURE and ADAPTIVE LOSS BALANCING.

Key Improvements:
1. Adaptive loss scaling to balance classification and regression tasks
2. Combined gradient updates (no alternating batches)
3. Gradient clipping for training stability
4. Larger shared layer capacity (64→32 neurons)
5. Learning rate scheduling
6. Comprehensive monitoring and diagnostics

Architecture: Shared (64→32) + Class (32→16→3) + Reg (32→16→1)
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

# Import utilities
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
# Results I/O and plotting utilities


class AdaptiveLossScaler:
    """
    Dynamically scale losses to keep them in similar magnitude ranges.
    This prevents one task from dominating training.
    """
    def __init__(self, alpha=0.9):
        self.alpha = alpha  # EMA factor
        self.class_loss_scale = 1.0
        self.reg_loss_scale = 1.0
        self.class_loss_history = []
        self.reg_loss_history = []
    
    def update(self, class_loss, reg_loss):
        """Update loss scales based on recent history."""
        self.class_loss_history.append(class_loss)
        self.reg_loss_history.append(reg_loss)
        
        # Keep only recent history
        if len(self.class_loss_history) > 100:
            self.class_loss_history = self.class_loss_history[-100:]
            self.reg_loss_history = self.reg_loss_history[-100:]
        
        # Calculate average magnitudes
        if len(self.class_loss_history) >= 10:
            avg_class = np.mean(self.class_loss_history[-10:])
            avg_reg = np.mean(self.reg_loss_history[-10:])
            
            # Update scales with EMA
            if avg_class > 0:
                self.class_loss_scale = self.alpha * self.class_loss_scale + (1 - self.alpha) * (1.0 / avg_class)
            if avg_reg > 0:
                self.reg_loss_scale = self.alpha * self.reg_loss_scale + (1 - self.alpha) * (1.0 / avg_reg)
    
    def scale_losses(self, class_loss, reg_loss, reg_weight=0.5):
        """Apply scaling to losses."""
        scaled_class = class_loss * self.class_loss_scale * (1 - reg_weight)
        scaled_reg = reg_loss * self.reg_loss_scale * reg_weight
        return scaled_class, scaled_reg


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_multitask_epoch_improved(model, wine_loader, ethanol_loader, device,
                                   classification_criterion, regression_criterion,
                                   optimizer, loss_scaler, reg_weight=0.5,
                                   max_grad_norm=1.0, warmup=False):
    """
    Improved training function with proper loss balancing and gradient combining.
    
    Key improvements:
    - Process both tasks in the same batch when possible
    - Adaptive loss scaling
    - Gradient clipping
    - Better logging
    """
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_reg_loss = 0
    total_scaled_class_loss = 0
    total_scaled_reg_loss = 0
    num_batches = 0
    
    # Create iterators
    wine_iter = iter(wine_loader)
    ethanol_iter = iter(ethanol_loader)
    
    # Process both datasets in parallel (not alternating!)
    max_batches = min(len(wine_loader), len(ethanol_loader))
    
    for _ in range(max_batches):
        optimizer.zero_grad()
        batch_class_loss = 0
        batch_reg_loss = 0
        
        # ====================================================================
        # WINE CLASSIFICATION TASK
        # ====================================================================
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
            
        except StopIteration:
            class_loss = torch.tensor(0.0, device=device)
            batch_class_loss = 0
        
        # ====================================================================
        # ETHANOL REGRESSION TASK
        # ====================================================================
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
            
        except StopIteration:
            reg_loss = torch.tensor(0.0, device=device)
            batch_reg_loss = 0
        
        # ====================================================================
        # IMPROVED ADAPTIVE LOSS SCALING WITH WARMUP
        # ====================================================================
        # Update scaler with current losses
        loss_scaler.update(batch_class_loss, batch_reg_loss)
        
        # Scale losses to similar magnitudes with warmup consideration
        if warmup:
            # During warmup phase, gradually increase loss scaling
            warmup_factor = 0.5  # Start with gentler scaling
            scaled_class = class_loss * warmup_factor
            scaled_reg = reg_loss * warmup_factor
        else:
            # Normal scaling after warmup
            scaled_class, scaled_reg = loss_scaler.scale_losses(
                class_loss, reg_loss, reg_weight
            )
        
        # Add L2 regularization if needed during non-warmup phase
        if not warmup:
            l2_lambda = 1e-5
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            l2_reg *= l2_lambda
        else:
            l2_reg = torch.tensor(0., device=device)
        
        # Combined loss with regularization
        batch_loss = scaled_class + scaled_reg + l2_reg
        
        # ====================================================================
        # BACKPROPAGATION WITH GRADIENT CLIPPING
        # ====================================================================
        batch_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        total_loss += batch_loss.item()
        total_class_loss += batch_class_loss
        total_reg_loss += batch_reg_loss
        total_scaled_class_loss += scaled_class.item()
        total_scaled_reg_loss += scaled_reg.item()
        num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_class_loss = total_class_loss / num_batches if num_batches > 0 else 0
    avg_reg_loss = total_reg_loss / num_batches if num_batches > 0 else 0
    avg_scaled_class = total_scaled_class_loss / num_batches if num_batches > 0 else 0
    avg_scaled_reg = total_scaled_reg_loss / num_batches if num_batches > 0 else 0
    
    return {
        'total_loss': avg_loss,
        'class_loss': avg_class_loss,
        'reg_loss': avg_reg_loss,
        'scaled_class_loss': avg_scaled_class,
        'scaled_reg_loss': avg_scaled_reg,
        'class_scale': loss_scaler.class_loss_scale,
        'reg_scale': loss_scaler.reg_loss_scale
    }


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


def plot_training_history_improved(history):
    """Plot comprehensive training history with loss scaling information."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'], label='Total Loss', linewidth=2, color='purple')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Combined Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Raw losses
    axes[0, 1].plot(history['class_loss'], label='Class Loss', color='red', linewidth=2)
    axes[0, 1].plot(history['reg_loss'], label='Reg Loss', color='teal', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Raw Task Losses', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scaled losses
    axes[0, 2].plot(history['scaled_class_loss'], label='Scaled Class', color='darkred', linewidth=2)
    axes[0, 2].plot(history['scaled_reg_loss'], label='Scaled Reg', color='darkcyan', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('Loss', fontsize=11)
    axes[0, 2].set_title('Scaled Task Losses (After Balancing)', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss scales
    axes[1, 0].plot(history['class_scale'], label='Class Scale', color='red', linewidth=2)
    axes[1, 0].plot(history['reg_scale'], label='Reg Scale', color='teal', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Scale Factor', fontsize=11)
    axes[1, 0].set_title('Adaptive Loss Scales', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio
    if len(history['class_loss']) > 0 and len(history['reg_loss']) > 0:
        loss_ratio = [c / (r + 1e-8) for c, r in zip(history['class_loss'], history['reg_loss'])]
        axes[1, 1].plot(loss_ratio, label='Class/Reg Ratio', color='orange', linewidth=2)
        axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', label='Equal (1:1)')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Ratio', fontsize=11)
        axes[1, 1].set_title('Loss Magnitude Ratio', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Training dynamics
    if 'eval_accuracy' in history and 'eval_r2' in history and len(history['eval_accuracy']) > 0:
        ax2 = axes[1, 2].twinx()
        eval_epochs = [i * 5 for i in range(len(history['eval_accuracy']))]  # Assuming EVAL_EVERY=5
        axes[1, 2].plot(eval_epochs, history['eval_accuracy'], label='Accuracy', color='blue', linewidth=2, marker='o')
        ax2.plot(eval_epochs, history['eval_r2'], label='R²', color='green', linewidth=2, marker='s')
        axes[1, 2].set_xlabel('Epoch', fontsize=11)
        axes[1, 2].set_ylabel('Accuracy', fontsize=11, color='blue')
        ax2.set_ylabel('R²', fontsize=11, color='green')
        axes[1, 2].set_title('Validation Metrics', fontsize=12, fontweight='bold')
        axes[1, 2].tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = axes[1, 2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='best')
    
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
        wine_output = model(wine_sample.to(device))
        ethanol_output = model(ethanol_sample.to(device))
    
    shared_data = wine_output['shared']
    
    if 'spk_rec1' in shared_data and 'spk_rec2' in shared_data:
        # Two shared layers architecture
        wine_shared1 = shared_data['spk_rec1'].cpu().numpy()
        wine_shared2 = shared_data['spk_rec2'].cpu().numpy()
        ethanol_shared1 = ethanol_output['shared']['spk_rec1'].cpu().numpy()
        ethanol_shared2 = ethanol_output['shared']['spk_rec2'].cpu().numpy()
        
        # Get task-specific hidden layers
        wine_class_hidden = wine_output['classification']['spk_rec'].cpu().numpy()
        wine_reg_hidden = wine_output['regression']['spk_rec'].cpu().numpy()
        ethanol_class_hidden = ethanol_output['classification']['spk_rec'].cpu().numpy()
        ethanol_reg_hidden = ethanol_output['regression']['spk_rec'].cpu().numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Wine - Shared Layer 1
        axes[0, 0].imshow(wine_shared1[:, 0, :].T, aspect='auto', cmap='binary')
        axes[0, 0].set_xlabel('Time Step', fontsize=10)
        axes[0, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 0].set_title('Wine: Shared Layer 1', fontsize=12, fontweight='bold')
        
        # Wine - Shared Layer 2
        axes[0, 1].imshow(wine_shared2[:, 0, :].T, aspect='auto', cmap='binary')
        axes[0, 1].set_xlabel('Time Step', fontsize=10)
        axes[0, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 1].set_title('Wine: Shared Layer 2', fontsize=12, fontweight='bold')
        
        # Wine - Classification Hidden
        axes[0, 2].imshow(wine_class_hidden[:, 0, :].T, aspect='auto', cmap='Reds')
        axes[0, 2].set_xlabel('Time Step', fontsize=10)
        axes[0, 2].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 2].set_title('Wine: Classification Hidden', fontsize=12, fontweight='bold')
        
        # Wine - Regression Hidden
        axes[0, 3].imshow(wine_reg_hidden[:, 0, :].T, aspect='auto', cmap='Blues')
        axes[0, 3].set_xlabel('Time Step', fontsize=10)
        axes[0, 3].set_ylabel('Neuron Index', fontsize=10)
        axes[0, 3].set_title('Wine: Regression Hidden', fontsize=12, fontweight='bold')
        
        # Ethanol - Shared Layer 1
        axes[1, 0].imshow(ethanol_shared1[:, 0, :].T, aspect='auto', cmap='binary')
        axes[1, 0].set_xlabel('Time Step', fontsize=10)
        axes[1, 0].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 0].set_title('Ethanol: Shared Layer 1', fontsize=12, fontweight='bold')
        
        # Ethanol - Shared Layer 2
        axes[1, 1].imshow(ethanol_shared2[:, 0, :].T, aspect='auto', cmap='binary')
        axes[1, 1].set_xlabel('Time Step', fontsize=10)
        axes[1, 1].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 1].set_title('Ethanol: Shared Layer 2', fontsize=12, fontweight='bold')
        
        # Ethanol - Classification Hidden
        axes[1, 2].imshow(ethanol_class_hidden[:, 0, :].T, aspect='auto', cmap='Reds')
        axes[1, 2].set_xlabel('Time Step', fontsize=10)
        axes[1, 2].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 2].set_title('Ethanol: Classification Hidden', fontsize=12, fontweight='bold')
        
        # Ethanol - Regression Hidden
        axes[1, 3].imshow(ethanol_reg_hidden[:, 0, :].T, aspect='auto', cmap='Blues')
        axes[1, 3].set_xlabel('Time Step', fontsize=10)
        axes[1, 3].set_ylabel('Neuron Index', fontsize=10)
        axes[1, 3].set_title('Ethanol: Regression Hidden', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.show()


def main(selected_arch_name='improved_shared_64_32_class_16_reg_16'):
    """Main training and evaluation pipeline with improved multitask learning."""
    print("="*80)
    print("IMPROVED MULTITASK SNN TRAINING")
    print("WITH ADAPTIVE LOSS BALANCING & GRADIENT CLIPPING")
    print("="*80)
    
    # Force use of improved architecture
    print(f"\n✓ Using architecture: {selected_arch_name}")
    
    # Load the architecture
    try:
        arch_module = load_architecture(selected_arch_name)
        MultitaskSNN = arch_module.MultitaskSNN
    except:
        print(f"\nERROR: Could not load architecture '{selected_arch_name}'")
        print("Make sure the improved_multitask_snn.py file is in the architectures folder!")
        sys.exit(1)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wine_data_path = os.path.join(base_dir, "data", "wine")
    ethanol_data_path = os.path.join(base_dir, "data", "wine")
    # Save trained models into shared directory 'tr'
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nData paths:")
    print(f"  Wine: {wine_data_path}")
    print(f"  Ethanol: {ethanol_data_path}")
    
    # ========================================================================
    # IMPROVED HYPERPARAMETERS
    # ========================================================================
    # Time series config
    SEQUENCE_LENGTH = 1000  # Longer sequences (from train_two_shared_layers.py)
    DOWNSAMPLE_FACTOR = 2   # Keep same factor
    ENCODING_TYPE = 'direct'
    
    # Model config - Match successful architecture
    BETA = 0.95             # Increased memory
    DROPOUT_RATE = 0.2      # Increased regularization
    SHARED_SIZE = 32        # First shared layer
    CLASS_HIDDEN = 16       # Classification branch
    REG_HIDDEN = 16        # Regression branch
    
    # Training config - IMPROVED
    BASE_LEARNING_RATE = 0.0005  # Start smaller for stability
    NUM_EPOCHS = 150              # Train longer for convergence
    BATCH_SIZE = 16               # Smaller batches 
    REG_WEIGHT = 0.5             # Equal task weighting
    MAX_GRAD_NORM = 0.5          # Tighter gradient clipping
    EVAL_EVERY = 5               # Evaluate every N epochs
    
    # Model save directory
    save_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    print(f"\nImproved Training Configuration:")
    print(f"  Base LR: {BASE_LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Gradient clipping: {MAX_GRAD_NORM}")
    print(f"  Adaptive loss scaling: Enabled")
    print(f"  Evaluation every: {EVAL_EVERY} epochs")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    
    # ========================================================================
    # LOAD AND PREPROCESS DATA
    # ========================================================================
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    wine_df = load_wine_dataset(wine_data_path)
    ethanol_df = load_ethanol_dataset(ethanol_data_path)
    
    if wine_df.empty or ethanol_df.empty:
        print("\nERROR: Data not found!")
        sys.exit(1)
    
    # Remove stabilization period
    print("\nRemoving first 500 time points from wine data...")
    wine_processed = []
    for filename in wine_df['Filename'].unique():
        file_data = wine_df[wine_df['Filename'] == filename].copy()
        if len(file_data) > 500:
            file_data = file_data.iloc[500:].reset_index(drop=True)
            wine_processed.append(file_data)
    wine_df = pd.concat(wine_processed, ignore_index=True) if wine_processed else pd.DataFrame()
    
    print("Removing first 500 time points from ethanol data...")
    ethanol_processed = []
    for filename in ethanol_df['Filename'].unique():
        file_data = ethanol_df[ethanol_df['Filename'] == filename].copy()
        if len(file_data) > 500:
            file_data = file_data.iloc[500:].reset_index(drop=True)
            ethanol_processed.append(file_data)
    ethanol_df = pd.concat(ethanol_processed, ignore_index=True) if ethanol_processed else pd.DataFrame()
    
    # Preprocess time series
    preprocessed_data = preprocess_time_series_features(
        wine_df, ethanol_df,
        sequence_length=SEQUENCE_LENGTH,
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    splits = create_train_test_split_sequences(
        preprocessed_data, test_size=0.2, random_state=42
    )
    
    # ========================================================================
    # SPIKE ENCODING
    # ========================================================================
    print("\n" + "="*80)
    print("SPIKE ENCODING")
    print("="*80)
    
    spike_encoded_data = encode_time_series_datasets(
        splits, encoding_type=ENCODING_TYPE, num_steps=None
    )
    
    # Extract data
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
    
    print(f"\n✓ Spike trains created")
    print(f"  Wine: {X_train_wine_spikes.shape}")
    print(f"  Ethanol: {X_train_ethanol_spikes.shape}")
    
    # Create dataloaders
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
    # CREATE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    input_size = splits['wine']['X_train'].shape[2]
    
    model = MultitaskSNN(
        input_size=input_size,
        num_classes=3,
        beta=BETA,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    print(model.get_architecture_summary())
    
    # ========================================================================
    # TRAINING SETUP - IMPROVED
    # ========================================================================
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Adaptive loss scaler
    loss_scaler = AdaptiveLossScaler(alpha=0.9)
    
    print("\n✓ Training setup complete")
    print(f"  Optimizer: Adam (LR={BASE_LEARNING_RATE})")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  Loss scaler: Adaptive EMA (alpha=0.9)")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    history = {
        'total_loss': [],
        'class_loss': [],
        'reg_loss': [],
        'scaled_class_loss': [],
        'scaled_reg_loss': [],
        'class_scale': [],
        'reg_scale': [],
        'eval_accuracy': [],
        'eval_r2': []
    }
    
    best_combined_metric = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    num_warmup_epochs = 10
    
    for epoch in range(NUM_EPOCHS):
        # Learning rate warmup
        if epoch < num_warmup_epochs:
            lr = BASE_LEARNING_RATE * ((epoch + 1) / num_warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Train
        train_metrics = train_multitask_epoch_improved(
            model, wine_train_loader, ethanol_train_loader, device,
            classification_criterion, regression_criterion,
            optimizer, loss_scaler, reg_weight=REG_WEIGHT,
            max_grad_norm=MAX_GRAD_NORM,
            warmup=(epoch < num_warmup_epochs)
        )
        
        # Log metrics
        history['total_loss'].append(train_metrics['total_loss'])
        history['class_loss'].append(train_metrics['class_loss'])
        history['reg_loss'].append(train_metrics['reg_loss'])
        history['scaled_class_loss'].append(train_metrics['scaled_class_loss'])
        history['scaled_reg_loss'].append(train_metrics['scaled_reg_loss'])
        history['class_scale'].append(train_metrics['class_scale'])
        history['reg_scale'].append(train_metrics['reg_scale'])
        
        # Evaluate periodically
        if (epoch + 1) % EVAL_EVERY == 0:
            class_results = evaluate_classification(model, wine_test_loader, device)
            reg_results = evaluate_regression(model, ethanol_test_loader, device)
            
            history['eval_accuracy'].append(class_results['accuracy'])
            history['eval_r2'].append(reg_results['r2'])
            
            # Calculate combined metric for model selection
            combined_metric = (class_results['accuracy'] + reg_results['r2']) / 2
            
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'combined_metric': combined_metric,
                    'class_accuracy': class_results['accuracy'],
                    'reg_r2': reg_results['r2']
                }, os.path.join(save_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch} epochs')
                    print(f'Best combined metric: {best_combined_metric:.4f} at epoch {best_epoch}')
                    break
            
            # Combined metric (you can adjust weights)
            combined_metric = 0.5 * class_results['accuracy'] + 0.5 * max(0, reg_results['r2'])
            
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_epoch = epoch + 1
            
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
            print(f"  Total Loss:     {train_metrics['total_loss']:.4f}")
            print(f"  Class Loss:     {train_metrics['class_loss']:.4f} (scale: {train_metrics['class_scale']:.3f})")
            print(f"  Reg Loss:       {train_metrics['reg_loss']:.4f} (scale: {train_metrics['reg_scale']:.3f})")
            print(f"  Accuracy:       {class_results['accuracy']:.4f}")
            print(f"  R²:             {reg_results['r2']:.4f}")
            print(f"  Combined:       {combined_metric:.4f} (best: {best_combined_metric:.4f} @ epoch {best_epoch})")
        
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {train_metrics['total_loss']:.4f} | "
                  f"Class: {train_metrics['class_loss']:.4f} | Reg: {train_metrics['reg_loss']:.4f}")
        
        # Update learning rate
        scheduler.step(train_metrics['total_loss'])
    
    print("\n✓ Training completed!")
    
    # ========================================================================
    # LOSS MAGNITUDE DIAGNOSTIC
    # ========================================================================
    print("\n" + "="*80)
    print("LOSS MAGNITUDE DIAGNOSTIC")
    print("="*80)
    
    with torch.no_grad():
        sample_wine = next(iter(wine_test_loader))
        sample_eth = next(iter(ethanol_test_loader))
        
        wine_out = model(sample_wine[0].to(device).permute(1, 0, 2))
        eth_out = model(sample_eth[0].to(device).permute(1, 0, 2))
        
        final_class_loss = classification_criterion(
            wine_out['classification']['mem_rec'].sum(0), 
            sample_wine[1].to(device).long()
        )
        final_reg_loss = regression_criterion(
            eth_out['regression']['mem_rec'].sum(0).squeeze(),
            sample_eth[1].to(device)
        )
        
        print(f"\nFinal raw loss magnitudes:")
        print(f"  Classification: {final_class_loss:.4f}")
        print(f"  Regression:     {final_reg_loss:.4f}")
        print(f"  Ratio (C/R):    {final_class_loss/final_reg_loss:.4f}")
        print(f"\nFinal loss scales:")
        print(f"  Class scale:    {loss_scaler.class_loss_scale:.4f}")
        print(f"  Reg scale:      {loss_scaler.reg_loss_scale:.4f}")
        print(f"\nLoss balance assessment:")
        ratio = final_class_loss / final_reg_loss
        if 0.5 <= ratio <= 2.0:
            print(f"  ✓ Losses are well balanced (ratio: {ratio:.2f})")
        elif ratio > 2.0:
            print(f"  ⚠ Classification loss dominates (ratio: {ratio:.2f})")
        else:
            print(f"  ⚠ Regression loss dominates (ratio: {ratio:.2f})")
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    class_results = evaluate_classification(model, wine_test_loader, device)
    reg_results = evaluate_regression(model, ethanol_test_loader, device)
    
    print("\n--- WINE CLASSIFICATION ---")
    print(f"  Accuracy:  {class_results['accuracy']:.4f}")
    print(f"  Precision: {class_results['precision']:.4f}")
    print(f"  Recall:    {class_results['recall']:.4f}")
    print(f"  F1-Score:  {class_results['f1']:.4f}")
    
    print("\n--- ETHANOL REGRESSION ---")
    print(f"  RMSE: {reg_results['rmse']:.4f}")
    print(f"  MAE:  {reg_results['mae']:.4f}")
    print(f"  R²:   {reg_results['r2']:.4f}")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (f"{selected_arch_name}_improved_"
                  f"acc-{class_results['accuracy']:.3f}_"
                  f"r2-{reg_results['r2']:.3f}_"
                  f"{timestamp}.pth")
    model_path = os.path.join(models_dir, model_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture_name': selected_arch_name,
        'architecture': {
            'input_size': input_size,
            'num_classes': 3,
            'beta': BETA,
            'dropout_rate': DROPOUT_RATE
        },
        'hyperparameters': {
            'learning_rate': BASE_LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'reg_weight': REG_WEIGHT,
            'max_grad_norm': MAX_GRAD_NORM,
            'sequence_length': SEQUENCE_LENGTH,
            'downsample_factor': DOWNSAMPLE_FACTOR,
            'encoding_type': ENCODING_TYPE
        },
        'results': {
            'classification': class_results,
            'regression': reg_results,
            'best_combined_metric': best_combined_metric,
            'best_epoch': best_epoch
        },
        'history': history,
        'scalers': {
            'wine_scaler': preprocessed_data['wine']['scaler_X'],
            'ethanol_scaler': preprocessed_data['ethanol']['scaler_X'],
            'ethanol_scaler_y': ethanol_scaler_y
        },
        'label_encoder': wine_label_encoder,
        'feature_names': splits['wine']['feature_names'],
        'loss_scaler_final': {
            'class_scale': loss_scaler.class_loss_scale,
            'reg_scale': loss_scaler.reg_loss_scale
        }
    }, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Save history json and generate static plots (useful for HPO and CI)
    json_path = os.path.join(models_dir, 'history.json')
    try:
        save_history_json(history, json_path)
        saved_plot_files = plot_history(history, models_dir)
        print(f"\nSaved training history JSON to: {json_path}")
        if saved_plot_files:
            print("Saved plots:")
            for p in saved_plot_files:
                print(f"  - {p}")
    except Exception as e:
        print(f"Warning: failed to save history or plots: {e}")

    # Training history with loss scaling (interactive / inline)
    print("\n1. Training history with adaptive loss scaling...")
    plot_training_history_improved(history)
    
    # Classification results
    print("2. Classification results...")
    class_names = wine_label_encoder.classes_
    plot_classification_results(class_results, class_names)
    
    # Regression results
    print("3. Regression results...")
    plot_regression_results(reg_results)
    
    # Shared layer activity visualization
    print("4. Shared layer activity...")
    wine_sample = X_test_wine_spikes[0:1, :, :].permute(1, 0, 2)
    ethanol_sample = X_test_ethanol_spikes[0:1, :, :].permute(1, 0, 2)
    plot_shared_layer_activity(model, wine_sample, ethanol_sample, device)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nArchitecture: {selected_arch_name}")
    print(f"\nImprovements Applied:")
    print(f"  ✓ Adaptive loss scaling (balanced {len(history['class_loss'])} epochs)")
    print(f"  ✓ Combined gradient updates (no alternating)")
    print(f"  ✓ Gradient clipping (max norm: {MAX_GRAD_NORM})")
    print(f"  ✓ Larger shared capacity (64→32 neurons)")
    print(f"  ✓ Learning rate scheduling")
    print(f"  ✓ Dropout regularization ({DROPOUT_RATE})")
    
    print(f"\nTime Series Config:")
    print(f"  - Sequence length: {SEQUENCE_LENGTH}")
    print(f"  - Downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"  - Encoding type: {ENCODING_TYPE}")
    print(f"  - Input features: {input_size}")
    
    print(f"\nFinal Performance:")
    print(f"  Wine Classification:")
    print(f"    - Accuracy: {class_results['accuracy']:.4f}")
    print(f"    - F1-Score: {class_results['f1']:.4f}")
    print(f"  Ethanol Regression:")
    print(f"    - RMSE: {reg_results['rmse']:.4f}")
    print(f"    - R²:   {reg_results['r2']:.4f}")
    print(f"  Combined Metric: {0.5 * class_results['accuracy'] + 0.5 * max(0, reg_results['r2']):.4f}")
    print(f"  Best Epoch: {best_epoch}")
    
    print(f"\nModel saved to: {model_path}")
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80)
    
    return model, history, class_results, reg_results


if __name__ == "__main__":
    # Force use of improved architecture
    if len(sys.argv) > 1:
        main(selected_arch_name=sys.argv[1])
    else:
        main(selected_arch_name='improved_shared_64_32_class_16_reg_16')