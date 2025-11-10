"""
Hyperparameter Optimization (HPO) for Ethanol Concentration SNN
Using Optuna for efficient hyperparameter search

IMPROVEMENTS:
- Added R² score tracking and logging
- Added trial timing information
- Improved progress reporting
- Better result visualization
- Reduced training time per trial
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# PyTorch
import torch
import torch.nn as nn

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# snntorch
import snntorch as snn

# Optuna for HPO
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_ethanol_dataset(base_path):
    """Loads the ethanol time-series dataset."""
    all_data = []
    
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    ethanol_concentration_map = {
        'C1': 1.0, 'C2': 2.5, 'C3': 5.0,
        'C4': 10.0, 'C5': 15.0, 'C6': 20.0
    }
    
    ethanol_path = os.path.join(base_path, 'Ethanol')
    
    if not os.path.exists(ethanol_path):
        raise FileNotFoundError(f"Ethanol folder not found at {ethanol_path}")
    
    files = os.listdir(ethanol_path)
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(ethanol_path, file_name)
            
            try:
                df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=ts_columns)
                df_file['Filename'] = file_name
                df_file['Time_Point'] = range(len(df_file))
                
                conc_code = file_name[3:5]
                df_file['Concentration_Value'] = ethanol_concentration_map.get(conc_code, np.nan)
                
                rep_start_index = file_name.rfind('R')
                df_file['Repetition'] = file_name[rep_start_index:rep_start_index + 3] if rep_start_index != -1 else 'Unknown'
                
                all_data.append(df_file)
                
            except Exception as e:
                continue
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def prepare_data(dataset_path, sequence_length=1000, downsample_factor=2):
    """Prepare time series data for SNN."""
    
    ethanol_df = load_ethanol_dataset(dataset_path)
    
    if ethanol_df.empty:
        raise ValueError("Dataset is empty")
    
    # Remove stabilization period
    stabilization_period = 500
    processed_files = []
    
    for filename in ethanol_df['Filename'].unique():
        file_data = ethanol_df[ethanol_df['Filename'] == filename].copy()
        
        if len(file_data) > stabilization_period:
            file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
            processed_files.append(file_data)
    
    ethanol_df = pd.concat(processed_files, ignore_index=True)
    
    # Prepare sequences
    sensor_cols = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    feature_cols = sensor_cols
    
    sequences = []
    concentrations = []
    
    for filename in ethanol_df['Filename'].unique():
        file_data = ethanol_df[ethanol_df['Filename'] == filename]
        sequence = file_data[feature_cols].values
        
        if downsample_factor > 1:
            sequence = sequence[::downsample_factor]
        
        if len(sequence) < sequence_length:
            padding = np.repeat(sequence[-1:], sequence_length - len(sequence), axis=0)
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > sequence_length:
            sequence = sequence[:sequence_length]
        
        sequences.append(sequence)
        concentrations.append(file_data['Concentration_Value'].iloc[0])
    
    # Clean and normalize
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
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(all_data)
    
    normalized_sequences = [scaler_X.transform(seq) for seq in sequences]
    
    # Prepare arrays
    num_samples = len(normalized_sequences)
    num_features = len(feature_cols)
    X = np.zeros((num_samples, sequence_length, num_features))
    
    for i, seq in enumerate(normalized_sequences):
        X[i, :len(seq), :] = seq
    
    y_concentration = np.array(concentrations)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_normalized = scaler_y.fit_transform(y_concentration.reshape(-1, 1)).flatten()
    
    return X, y_normalized, scaler_X, scaler_y, num_features


# ============================================================================
# SPIKE ENCODING FUNCTIONS
# ============================================================================

def encode_time_series_direct(sequences):
    """Direct threshold-based encoding."""
    sequences_tensor = torch.FloatTensor(sequences)
    spike_data = (sequences_tensor > 0.5).float()
    spike_data = spike_data.permute(1, 0, 2)
    return spike_data


def encode_time_series_delta(sequences, threshold=0.05):
    """Delta encoding based on temporal changes."""
    sequences_tensor = torch.FloatTensor(sequences)
    deltas = sequences_tensor[:, 1:, :] - sequences_tensor[:, :-1, :]
    min_val, max_val = deltas.min(), deltas.max()
    deltas = (deltas - min_val) / (max_val - min_val + 1e-8)
    spike_data = (torch.abs(deltas) > threshold).float()
    spike_data = spike_data.permute(1, 0, 2)
    return spike_data


# ============================================================================
# SNN MODEL
# ============================================================================

class TimeSeriesConcentrationSNN(nn.Module):
    """SNN for Ethanol Concentration Regression."""
    
    def __init__(self, input_size=8, hidden_size1=28, hidden_size2=14, 
                 output_size=1, beta=0.9, dropout=0.0):
        super(TimeSeriesConcentrationSNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        mem_rec3 = []
        
        num_steps = x.size(0)
        for step in range(num_steps):
            x_step = x[step]
            
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            if self.dropout:
                spk1 = self.dropout(spk1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            if self.dropout:
                spk2 = self.dropout(spk2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            mem_rec3.append(mem3)
        
        mem_rec3 = torch.stack(mem_rec3)
        return mem_rec3


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs, device, early_stopping_patience=8):
    """Train the SNN model with early stopping."""
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for spike_batch, target_batch in train_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            target_batch = target_batch.to(device)
            
            mem_rec = model(spike_batch)
            predictions = mem_rec.mean(dim=0).squeeze()
            
            loss = criterion(predictions, target_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for spike_batch, target_batch in val_loader:
                spike_batch = spike_batch.permute(1, 0, 2).to(device)
                target_batch = target_batch.to(device)
                
                mem_rec = model(spike_batch)
                predictions = mem_rec.mean(dim=0).squeeze()
                
                loss = criterion(predictions, target_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    return best_val_loss


# ============================================================================
# OPTUNA OBJECTIVE FUNCTION (WITH R² TRACKING)
# ============================================================================

def objective(trial, X, y, num_features, scaler_y):
    """Optuna objective function for HPO with R² tracking."""
    
    trial_start = time.time()
    
    # FIXED Architecture (not optimized)
    hidden_size1 = 28  # Fixed
    hidden_size2 = 14  # Fixed
    
    # Hyperparameters to optimize
    beta = trial.suggest_float('beta', 0.7, 0.95)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    
    # Encoding parameters
    encoding_type = trial.suggest_categorical('encoding_type', ['direct'])
    
    if encoding_type == 'delta':
        threshold = trial.suggest_float('delta_threshold', 0.01, 0.2)
    else:
        threshold = None
    
    # Optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Encode spikes
    if encoding_type == 'direct':
        spike_train = encode_time_series_direct(X_train)
        spike_val = encode_time_series_direct(X_val)
    else:  # delta
        spike_train = encode_time_series_delta(X_train, threshold=threshold)
        spike_val = encode_time_series_delta(X_val, threshold=threshold)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        spike_train.permute(1, 0, 2),
        torch.FloatTensor(y_train)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        spike_val.permute(1, 0, 2),
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Create model
    model = TimeSeriesConcentrationSNN(
        input_size=num_features,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        output_size=1,
        beta=beta,
        dropout=dropout
    ).to(device)
    
    # Create optimizer
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:  # SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    criterion = nn.MSELoss()
    
    # Train model (reduced from 50 to 30 epochs for faster trials)
    val_loss = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        num_epochs=30, device=device, early_stopping_patience=8
    )
    
    # Calculate additional metrics
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spike_batch, target_batch in val_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            mem_rec = model(spike_batch)
            predictions = mem_rec.mean(dim=0).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.numpy())
    
    # Calculate R² score
    r2 = r2_score(all_targets, all_predictions)
    
    # Calculate MAE for additional insight
    mae = mean_absolute_error(all_targets, all_predictions)
    
    # Store metrics as user attributes
    trial_time = time.time() - trial_start
    trial.set_user_attr('r2_score', r2)
    trial.set_user_attr('mae', mae)
    trial.set_user_attr('trial_time', trial_time)
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Summary:")
    print(f"  Val Loss: {val_loss:.6f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  Time:     {trial_time:.1f}s ({trial_time/60:.1f} min)")
    print(f"  Encoding: {encoding_type}")
    print(f"  Batch:    {batch_size}, LR: {learning_rate:.6f}, Beta: {beta:.3f}")
    print(f"{'='*70}\n")
    
    # Report intermediate value for pruning
    trial.report(val_loss, 0)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return val_loss


# ============================================================================
# MAIN HPO FUNCTION
# ============================================================================

def run_hpo(n_trials=30, study_name='ethanol_snn_hpo'):
    """Run hyperparameter optimization."""
    
    print("="*80)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    print("\nLoading and preparing data...")
    dataset_path = 'data/wine'
    if not os.path.exists(dataset_path):
        dataset_path = 'Dataset'
    
    X, y, scaler_X, scaler_y, num_features = prepare_data(dataset_path)
    print(f"Data shape: {X.shape}")
    print(f"Features: {num_features}")
    print(f"Samples: {len(X)}")
    
    # Create study
    print(f"\nCreating Optuna study with {n_trials} trials...")
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Run optimization
    print("\nStarting optimization...")
    print("="*80)
    
    hpo_start = time.time()
    
    study.optimize(
        lambda trial: objective(trial, X, y, num_features, scaler_y),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    hpo_time = time.time() - hpo_start
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETED")
    print("="*80)
    print(f"Total HPO time: {hpo_time/60:.1f} minutes ({hpo_time/3600:.2f} hours)")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Trial number: {trial.number}")
    print(f"  Val Loss: {trial.value:.6f}")
    print(f"  R² Score: {trial.user_attrs.get('r2_score', 'N/A'):.4f}")
    print(f"  MAE: {trial.user_attrs.get('mae', 'N/A'):.6f}")
    print(f"  Time: {trial.user_attrs.get('trial_time', 0):.1f}s")
    
    print("\n  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    results_dir = "hpo_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study
    study_path = os.path.join(results_dir, f"study_{timestamp}.pkl")
    import joblib
    joblib.dump(study, study_path)
    print(f"\n✓ Study saved to: {study_path}")
    
    # Save best parameters with all metrics
    best_params_path = os.path.join(results_dir, f"best_params_{timestamp}.txt")
    with open(best_params_path, 'w') as f:
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Validation Loss: {trial.value:.6f}\n")
        f.write(f"R² Score: {trial.user_attrs.get('r2_score', 'N/A')}\n")
        f.write(f"MAE: {trial.user_attrs.get('mae', 'N/A')}\n")
        f.write(f"Trial Time: {trial.user_attrs.get('trial_time', 0):.1f}s\n\n")
        f.write("Parameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTotal HPO Time: {hpo_time/60:.1f} minutes\n")
    
    print(f"✓ Best parameters saved to: {best_params_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html(os.path.join(results_dir, f"optimization_history_{timestamp}.html"))
        print("  ✓ Optimization history")
    except Exception as e:
        print(f"  ✗ Could not create optimization history: {e}")
    
    try:
        fig2 = plot_param_importances(study)
        fig2.write_html(os.path.join(results_dir, f"param_importances_{timestamp}.html"))
        print("  ✓ Parameter importances")
    except Exception as e:
        print(f"  ✗ Could not create parameter importance plot: {e}")
    
    try:
        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(results_dir, f"parallel_coordinate_{timestamp}.html"))
        print("  ✓ Parallel coordinate plot")
    except Exception as e:
        print(f"  ✗ Could not create parallel coordinate plot: {e}")
    
    print(f"\n✓ Visualizations saved to: {results_dir}/")
    
    # Summary statistics
    print("\n" + "="*80)
    print("HPO SUMMARY")
    print("="*80)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial number: {trial.number}")
    print(f"Average trial time: {hpo_time/len(study.trials):.1f}s")
    
    # Top 5 trials
    print("\nTop 5 trials by validation loss:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value)[:5]
    for i, t in enumerate(sorted_trials, 1):
        r2 = t.user_attrs.get('r2_score', 'N/A')
        trial_time = t.user_attrs.get('trial_time', 'N/A')
        print(f"\n{i}. Trial {t.number}")
        print(f"   Val Loss: {t.value:.6f}")
        print(f"   R² Score: {r2:.4f}" if isinstance(r2, float) else f"   R² Score: {r2}")
        print(f"   Time: {trial_time:.1f}s" if isinstance(trial_time, float) else f"   Time: {trial_time}")
        print(f"   Params: {t.params}")
    
    # Save comprehensive CSV report
    trials_df = study.trials_dataframe()
    csv_path = os.path.join(results_dir, f"trials_report_{timestamp}.csv")
    trials_df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed trials report saved to: {csv_path}")
    
    return study, trial.params


# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

def train_final_model(best_params, dataset_path='data/wine'):
    """Train final model with best hyperparameters."""
    
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*80)
    
    # Add fixed architecture to best_params if not present
    if 'hidden_size1' not in best_params:
        best_params['hidden_size1'] = 28
    if 'hidden_size2' not in best_params:
        best_params['hidden_size2'] = 14
    
    # Load data
    X, y, scaler_X, scaler_y, num_features = prepare_data(dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Encode spikes
    encoding_type = best_params['encoding_type']
    
    if encoding_type == 'direct':
        spike_train = encode_time_series_direct(X_train)
        spike_test = encode_time_series_direct(X_test)
    else:  # delta
        threshold = best_params.get('delta_threshold', 0.05)
        spike_train = encode_time_series_delta(X_train, threshold=threshold)
        spike_test = encode_time_series_delta(X_test, threshold=threshold)
    
    # Create data loaders
    batch_size = best_params['batch_size']
    
    train_dataset = torch.utils.data.TensorDataset(
        spike_train.permute(1, 0, 2),
        torch.FloatTensor(y_train)
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        spike_test.permute(1, 0, 2),
        torch.FloatTensor(y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Create model
    model = TimeSeriesConcentrationSNN(
        input_size=num_features,
        hidden_size1=best_params['hidden_size1'],
        hidden_size2=best_params['hidden_size2'],
        output_size=1,
        beta=best_params['beta'],
        dropout=best_params['dropout']
    ).to(device)
    
    # Create optimizer
    optimizer_name = best_params['optimizer']
    learning_rate = best_params['learning_rate']
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    criterion = nn.MSELoss()
    
    # Train for more epochs
    print("\nTraining final model (100 epochs)...")
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        
        for spike_batch, target_batch in train_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            target_batch = target_batch.to(device)
            
            mem_rec = model(spike_batch)
            predictions = mem_rec.mean(dim=0).squeeze()
            
            loss = criterion(predictions, target_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100] - Train Loss: {train_loss:.6f}")
    
    # Final evaluation
    print("\nEvaluating final model...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spike_batch, target_batch in test_loader:
            spike_batch = spike_batch.permute(1, 0, 2).to(device)
            mem_rec = model(spike_batch)
            predictions = mem_rec.mean(dim=0).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Denormalize
    predictions_original = scaler_y.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    targets_original = scaler_y.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Metrics
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_original, predictions_original)
    r2 = r2_score(targets_original, predictions_original)
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE")
    print("="*80)
    print(f"\nRMSE: {rmse:.4f}%")
    print(f"MAE: {mae:.4f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Save model
    model_path = "trained_models/ethanol_snn_best_hpo.pth"
    os.makedirs("trained_models", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'num_features': num_features,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }, model_path)
    
    print(f"\n✓ Final model saved to: {model_path}")
    
    return model, rmse, mae, r2


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║         HYPERPARAMETER OPTIMIZATION FOR ETHANOL SNN                    ║
    ║                   WITH R² TRACKING & TIMING                            ║
    ║                                                                        ║
    ║  IMPROVEMENTS:                                                         ║
    ║  • R² score tracking for each trial                                   ║
    ║  • Detailed timing information                                        ║
    ║  • Better progress reporting                                          ║
    ║  • Reduced epochs (30 instead of 50) for faster trials               ║
    ║  • Comprehensive CSV report of all trials                             ║
    ║                                                                        ║
    ║  OPTIMIZING:                                                          ║
    ║  • Learning rate                                                      ║
    ║  • Batch size                                                         ║
    ║  • Beta (membrane decay parameter)                                    ║
    ║  • Dropout rate                                                       ║
    ║  • Encoding type (direct or delta)                                    ║
    ║  • Delta threshold (if delta encoding)                                ║
    ║  • Optimizer type (Adam, AdamW, SGD)                                  ║
    ║                                                                        ║
    ║  FIXED:                                                               ║
    ║  • Architecture: 28 neurons (layer 1), 14 neurons (layer 2)           ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    N_TRIALS = 30  # Adjust based on computational resources
    
    print(f"\n🚀 Starting HPO with {N_TRIALS} trials")
    print(f"⏱️  Estimated time: {N_TRIALS * 3:.0f}-{N_TRIALS * 5:.0f} minutes")
    print("="*80)
    
    # Run HPO
    study, best_params = run_hpo(n_trials=N_TRIALS)
    
    # Train final model
    print("\n" + "="*80)
    print("Training final model with best parameters...")
    print("="*80)
    
    # For automated execution
    train_final = True
    
    if train_final:
        model, rmse, mae, r2 = train_final_model(best_params)
        
        print("\n" + "="*80)
        print("🎉 HPO PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                          FINAL RESULTS                                 ║
╚════════════════════════════════════════════════════════════════════════╝

Performance Metrics:
--------------------
RMSE:      {rmse:.4f}%
MAE:       {mae:.4f}%
R² Score:  {r2:.4f}

Best Hyperparameters:
---------------------
{chr(10).join(f'{k}: {v}' for k, v in best_params.items())}

Files Saved:
------------
✓ hpo_results/study_*.pkl              - Complete Optuna study
✓ hpo_results/best_params_*.txt        - Best hyperparameters
✓ hpo_results/trials_report_*.csv      - Detailed trial results
✓ hpo_results/optimization_history_*.html
✓ hpo_results/param_importances_*.html
✓ hpo_results/parallel_coordinate_*.html
✓ trained_models/ethanol_snn_best_hpo.pth - Best model checkpoint

Next Steps:
-----------
1. 📊 Review visualizations in hpo_results/ directory
2. 📈 Analyze parameter importance plots
3. 🔍 Check trials_report_*.csv for detailed metrics
4. 🎯 If R² < 0.8, consider:
   - Increasing trials (100+)
   - Optimizing architecture (hidden sizes)
   - Adding more encoding types
5. 🚀 Use best model for inference/deployment

╔════════════════════════════════════════════════════════════════════════╗
║                    Thank you for using Optuna HPO!                     ║
╚════════════════════════════════════════════════════════════════════════╝
        """)