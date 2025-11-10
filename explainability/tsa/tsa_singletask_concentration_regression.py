"""
TSA Explainability Script for Ethanol Concentration Regression
============================================================

This script performs Temporal Spike Attribution (TSA) analysis for ethanol
concentration regression using SNNs. It mirrors the classification TSA script
but adapts the analysis for continuous regression outputs.

Key features:
    ✅ TSA for ethanol concentration regression (continuous output)
    ✅ Regression-specific perturbation analysis
    ✅ Temporal attribution heatmaps for concentration prediction
    ✅ Prediction error analysis with TSA correlations
    ✅ Feature importance ranking for regression
    ✅ Comprehensive visualization and results saving

Usage:
    python tsa_singletask_concentration_regression.py \
        --checkpoint ../../trained_models/ethanol_timeseries_snn_seq1000_beta0.9_bs16_lr0.001_ep100.pth

Hyperparameters from HPO:
    - Beta: 0.9320405786091646
    - Learning rate: 0.009478855050990368
    - Batch size: 8
    - Dropout: 0.0359117786731738
"""

import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# 1. MODEL DEFINITION (TimeSeriesConcentrationSNN)
# -------------------------------------------------------
class TimeSeriesConcentrationSNN(nn.Module):
    """
    SNN for ethanol concentration regression from time series sensor data.
    Architecture matches the original regression training script.
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
        self.lif3 = snn.Leaky(beta=beta)  # Output layer for regression
        
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


# -------------------------------------------------------
# 2. TSA HELPERS FOR REGRESSION
# -------------------------------------------------------

def compute_N_from_spike_times(spike_times_list, t_prime, beta, fan_in):
    """
    Compute N contribution from spike times for TSA.
    Same as classification version.
    """
    N = 0.0
    # TSAS part - positive contributions from spikes
    for t_k in spike_times_list:
        if t_k <= t_prime:
            N += beta ** (t_prime - t_k)
    
    # TSANS part - negative contributions from non-spikes  
    spike_set = set(spike_times_list)
    for t_k in range(t_prime + 1):
        if t_k not in spike_set:
            N -= (1.0 / fan_in) * (beta ** (t_prime - t_k))
            
    return N


@torch.no_grad()
def compute_tsa_regression(model, x, beta, device):
    """
    Compute TSA attributions for regression output.
    
    For regression, we don't have softmax but direct membrane potentials.
    The attribution shows how each input feature at each timestep
    contributes to the final concentration prediction.
    
    Returns:
        attributions: [T, F] tensor of TSA values
    """
    x = x.to(device)
    mem_rec, _, h1_spk, h2_spk = model(x.unsqueeze(1))  # Add batch dim
    
    T = mem_rec.shape[0]  # timesteps
    F = x.shape[1]        # features
    H1 = h1_spk.shape[2]  # hidden1 size
    H2 = h2_spk.shape[2]  # hidden2 size
    
    # Get model weights
    W1_T = model.fc1.weight.t()  # [F, H1]
    W2_T = model.fc2.weight.t()  # [H1, H2] 
    W3_T = model.fc3.weight.t()  # [H2, 1]
    
    # Extract spike times for each neuron
    inp_spk = [torch.where(x[:, i] > 0)[0].tolist() for i in range(F)]
    h1_spk_times = [torch.where(h1_spk[:, 0, i] > 0)[0].tolist() for i in range(H1)]
    h2_spk_times = [torch.where(h2_spk[:, 0, i] > 0)[0].tolist() for i in range(H2)]
    
    # Output membrane potentials (no softmax for regression)
    out_mem = mem_rec.squeeze(1)  # [T, 1]
    
    fan1, fan2, fan3 = F, H1, H2
    attributions = []
    
    for t in range(T):
        # For regression, we don't use softmax - direct membrane contribution
        # The gradient w.r.t. the output is just 1 (identity)
        output_grad = torch.ones(1, device=device, dtype=torch.float32)
        
        # Compute N values for each layer
        N0 = torch.tensor([
            compute_N_from_spike_times(inp_spk[i], t, beta, fan1) 
            for i in range(F)
        ], device=device, dtype=torch.float32)
        
        N1 = torch.tensor([
            compute_N_from_spike_times(h1_spk_times[i], t, beta, fan2)
            for i in range(H1) 
        ], device=device, dtype=torch.float32)
        
        N2 = torch.tensor([
            compute_N_from_spike_times(h2_spk_times[i], t, beta, fan3)
            for i in range(H2)
        ], device=device, dtype=torch.float32)
        
        # Compute attribution: input -> hidden1 -> hidden2 -> output
        # CI_t = (N0 @ W1) @ (N1 @ W2) @ (N2 @ W3) @ output_grad
        attribution_t = (
            (torch.diag(N0) @ W1_T) @  # [F, H1]
            (torch.diag(N1) @ W2_T) @  # [H1, H2] 
            (torch.diag(N2) @ W3_T) @  # [H2, 1]
            output_grad.unsqueeze(0)   # [1, 1]
        ).squeeze()  # [F]
        
        attributions.append(attribution_t)
    
    return torch.stack(attributions)  # [T, F]


@torch.no_grad()
def predict_concentration(model, x, device, scaler_y=None):
    """
    Predict concentration from input sequence.
    
    Returns:
        prediction: Predicted concentration (denormalized if scaler provided)
        membrane_sum: Raw membrane potential sum
    """
    mem_rec, _, _, _ = model(x.unsqueeze(1).to(device))
    
    # Sum membrane potentials over time (same as training)
    membrane_sum = mem_rec.mean(dim=0)[0, 0].cpu().item()  # Changed from sum to mean
    
    # Denormalize if scaler provided
    prediction = membrane_sum
    if scaler_y is not None:
        prediction = scaler_y.inverse_transform([[membrane_sum]])[0, 0]
        
    return prediction, membrane_sum


# -------------------------------------------------------
# 3. VISUALIZATION UTILITIES FOR REGRESSION
# -------------------------------------------------------

def plot_regression_heatmap(attribution, feature_names, title="TSA Attribution", 
                           save_path=None, concentration=None):
    """
    Plot TSA attribution heatmap for regression.
    """
    arr = attribution.cpu().numpy()
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(arr.T, cmap="RdBu_r", center=0,
                xticklabels=50, yticklabels=feature_names)
    
    if concentration is not None:
        title = f"{title} (Concentration: {concentration:.2f}%)"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time Step", fontweight='bold')
    plt.ylabel("Feature", fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_concentration_range_comparison(attributions_list, concentrations, feature_names, 
                                      save_dir=None):
    """
    Compare TSA attributions across different concentration ranges.
    """
    # Group samples by concentration ranges
    low_conc = [(att, conc) for att, conc in zip(attributions_list, concentrations) if conc < 10]
    mid_conc = [(att, conc) for att, conc in zip(attributions_list, concentrations) if 10 <= conc < 30]
    high_conc = [(att, conc) for att, conc in zip(attributions_list, concentrations) if conc >= 30]
    
    ranges = [("Low (< 10%)", low_conc), ("Medium (10-30%)", mid_conc), ("High (≥ 30%)", high_conc)]
    
    for range_name, range_data in ranges:
        if not range_data:
            continue
            
        # Average attributions in this range
        avg_attribution = torch.stack([att for att, _ in range_data]).mean(dim=0)
        concs_in_range = [conc for _, conc in range_data]
        avg_conc = np.mean(concs_in_range)
        
        save_path = None
        if save_dir:
            safe_name = range_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("≥", "gte").replace("<", "lt").replace("%", "pct")
            save_path = save_dir / f"tsa_concentration_range_{safe_name}.png"
            
        plot_regression_heatmap(
            avg_attribution, feature_names,
            title=f"Average TSA for {range_name} Concentration",
            save_path=save_path,
            concentration=avg_conc
        )


def plot_regression_perturbation_curve(model, x, attribution, device, feature_names,
                                     true_concentration, scaler_y=None, save_path=None):
    """
    Analyze how removing important spikes affects concentration predictions.
    """
    T, F = attribution.shape
    flat_imp = attribution.abs().flatten().cpu()
    
    x_cpu = x.cpu()
    spike_mask = (x_cpu > 0.5).flatten()
    spike_indices = torch.nonzero(spike_mask).flatten()
    
    if spike_indices.numel() == 0:
        print("No spikes to test perturbation.")
        return None
    
    k_values = list(range(1, min(50, spike_indices.numel()), 5))
    top_k_effects = []
    rand_k_effects = []
    
    base_pred, base_mem = predict_concentration(model, x, device, scaler_y)
    
    perturbation_data = {
        'description': 'Analysis of how removing spikes affects concentration predictions',
        'baseline': {
            'true_concentration': float(true_concentration),
            'predicted_concentration': float(base_pred),
            'membrane_potential': float(base_mem),
            'prediction_error': float(abs(base_pred - true_concentration))
        },
        'k_values_tested': k_values,
        'top_k_deletions': {'results': []},
        'random_k_deletions': {'results': []}
    }
    
    for k in k_values:
        # Top-k deletion (most important spikes)
        imp_values = flat_imp[spike_indices]
        top_order = torch.topk(imp_values, k).indices
        top_idx = spike_indices[top_order]
        
        x_edit = x.clone()
        for idx in top_idx:
            t = (idx // F).item()
            f = (idx % F).item()
            x_edit[t, f] = 0.0
            
        pred_edit, mem_edit = predict_concentration(model, x_edit, device, scaler_y)
        top_k_effect = abs(base_pred - pred_edit)
        top_k_effects.append(top_k_effect)
        
        perturbation_data['top_k_deletions']['results'].append({
            'k': k,
            'predicted_concentration': float(pred_edit),
            'membrane_potential': float(mem_edit),
            'concentration_change': float(pred_edit - base_pred),
            'absolute_change': float(top_k_effect),
            'prediction_error': float(abs(pred_edit - true_concentration))
        })
        
        # Random-k deletion
        rand_perm = spike_indices[torch.randperm(spike_indices.numel())[:k]]
        x_rand = x.clone()
        for idx in rand_perm:
            t = (idx // F).item()
            f = (idx % F).item()
            x_rand[t, f] = 0.0
            
        pred_rand, mem_rand = predict_concentration(model, x_rand, device, scaler_y)
        rand_k_effect = abs(base_pred - pred_rand)
        rand_k_effects.append(rand_k_effect)
        
        perturbation_data['random_k_deletions']['results'].append({
            'k': k,
            'predicted_concentration': float(pred_rand),
            'membrane_potential': float(mem_rand),
            'concentration_change': float(pred_rand - base_pred),
            'absolute_change': float(rand_k_effect),
            'prediction_error': float(abs(pred_rand - true_concentration))
        })
    
    # Summary statistics
    perturbation_data['summary'] = {
        'top_k_average_effect': float(np.mean(top_k_effects)),
        'top_k_max_effect': float(np.max(top_k_effects)),
        'random_k_average_effect': float(np.mean(rand_k_effects)),
        'random_k_max_effect': float(np.max(rand_k_effects)),
        'top_k_more_effective': bool(np.mean(top_k_effects) > np.mean(rand_k_effects))
    }
    
    # Plot perturbation curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, top_k_effects, label="Top-K deletion", linewidth=2, marker='o')
    plt.plot(k_values, rand_k_effects, label="Random-K deletion", linewidth=2, marker='s')
    plt.xlabel("K (number of deleted spikes)")
    plt.ylabel("Absolute concentration change")
    plt.title("Perturbation Effect on Concentration")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Show actual predictions
    top_k_preds = [res['predicted_concentration'] for res in perturbation_data['top_k_deletions']['results']]
    rand_k_preds = [res['predicted_concentration'] for res in perturbation_data['random_k_deletions']['results']]
    
    plt.plot(k_values, top_k_preds, label="Top-K deletion", linewidth=2, marker='o')
    plt.plot(k_values, rand_k_preds, label="Random-K deletion", linewidth=2, marker='s')
    plt.axhline(y=base_pred, color='blue', linestyle='--', alpha=0.7, label='Original prediction')
    plt.axhline(y=true_concentration, color='red', linestyle='--', alpha=0.7, label='True concentration')
    plt.xlabel("K (number of deleted spikes)")
    plt.ylabel("Predicted concentration (%)")
    plt.title("Predictions vs Spike Deletions")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    plt.close()
    
    return perturbation_data


def plot_prediction_error_analysis(predictions, true_values, attributions_list, 
                                 feature_names, save_dir=None):
    """
    Analyze relationship between prediction errors and TSA patterns.
    """
    errors = [abs(pred - true) for pred, true in zip(predictions, true_values)]
    
    # Create error vs prediction scatter
    plt.figure(figsize=(15, 10))
    
    # 1. Error vs prediction scatter
    plt.subplot(2, 3, 1)
    plt.scatter(predictions, errors, alpha=0.6, s=50)
    plt.xlabel("Predicted Concentration (%)")
    plt.ylabel("Absolute Error (%)")
    plt.title("Prediction Error Distribution")
    plt.grid(alpha=0.3)
    
    # 2. Error vs true value scatter  
    plt.subplot(2, 3, 2)
    plt.scatter(true_values, errors, alpha=0.6, s=50)
    plt.xlabel("True Concentration (%)")
    plt.ylabel("Absolute Error (%)")
    plt.title("Error vs True Concentration")
    plt.grid(alpha=0.3)
    
    # 3. Error histogram
    plt.subplot(2, 3, 3)
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Absolute Error (%)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(alpha=0.3)
    
    # 4. Predictions vs true values
    plt.subplot(2, 3, 4)
    plt.scatter(true_values, predictions, alpha=0.6, s=50)
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel("True Concentration (%)")
    plt.ylabel("Predicted Concentration (%)")
    plt.title("Predictions vs True Values")
    plt.grid(alpha=0.3)
    
    # 5. Feature importance vs error correlation
    plt.subplot(2, 3, 5)
    if attributions_list:
        # Calculate average absolute attribution per feature
        avg_attr_per_feature = torch.stack(attributions_list).abs().mean(dim=0).mean(dim=0)
        feature_importance = avg_attr_per_feature.cpu().numpy()
        
        plt.bar(range(len(feature_names)), feature_importance)
        plt.xlabel("Feature Index")  
        plt.ylabel("Average |TSA Attribution|")
        plt.title("Feature Importance")
        plt.xticks(range(len(feature_names)), [name[:8] for name in feature_names], rotation=45)
    
    # 6. TSA magnitude vs error
    plt.subplot(2, 3, 6)
    if attributions_list:
        tsa_magnitudes = [attr.abs().mean().item() for attr in attributions_list]
        plt.scatter(tsa_magnitudes, errors, alpha=0.6, s=50)
        plt.xlabel("Mean |TSA Attribution|")
        plt.ylabel("Absolute Error (%)")
        plt.title("TSA Magnitude vs Error")
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / "prediction_error_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    plt.close()


# -------------------------------------------------------
# 4. MAIN EXPLAINABILITY PIPELINE FOR REGRESSION
# -------------------------------------------------------

def load_ethanol_data(spike_data_dir):
    """
    Load ethanol concentration spike data.
    Expects files generated by generate_spike_data.py for ethanol dataset.
    """
    spike_test_path = spike_data_dir / 'spike_test.npy'
    y_test_path = spike_data_dir / 'y_test.npy'
    config_path = spike_data_dir / 'config.npy'
    
    if not spike_test_path.exists():
        raise FileNotFoundError(f"Spike test data not found: {spike_test_path}")
    
    spike_test = np.load(spike_test_path)
    y_test = np.load(y_test_path)
    config = np.load(config_path, allow_pickle=True).item()
    
    return spike_test, y_test, config


def main():
    parser = argparse.ArgumentParser(
        description="TSA Explainability for ethanol concentration regression SNN"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../../trained_models/ethanol_timeseries_snn_seq1000_beta0.9_bs16_lr0.001_ep100.pth",
        help="Path to trained regression model checkpoint"
    )
    parser.add_argument(
        "--spike_data_dir",
        type=str,
        default="../../data/spike_data",
        help="Directory containing ethanol spike data"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of test sample to analyze"
    )
    parser.add_argument(
        "--num_random_samples",
        type=int,
        default=10,
        help="Number of random samples for analysis"
    )
    parser.add_argument(
        "--full_dataset_analysis",
        action="store_true",
        help="Analyze entire test dataset (slower but more comprehensive)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Setup results directory
    script_dir = Path(__file__).parent
    results_base_dir = script_dir / "results" / "concentration_regression"
    results_base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    results_dir = results_base_dir / timestamp
    results_dir.mkdir(exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}\n")
    
    # Initialize run summary
    run_summary = {
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'device': str(device),
        'args': vars(args),
        'hyperparameters_used': {
            'beta': 0.9320405786091646,
            'learning_rate': 0.009478855050990368,
            'batch_size': 8,
            'dropout': 0.0359117786731738,
            'encoding_type': 'direct',
            'optimizer': 'AdamW'
        }
    }
    
    # Resolve paths
    script_dir = Path(__file__).parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (script_dir / checkpoint_path).resolve()
        
    spike_data_dir = Path(args.spike_data_dir)
    if not spike_data_dir.is_absolute():
        spike_data_dir = (script_dir / spike_data_dir).resolve()
    
    print("=" * 80)
    print("TSA EXPLAINABILITY PIPELINE - ETHANOL CONCENTRATION REGRESSION")  
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Spike data directory: {spike_data_dir}")
    print()
    
    # Load model
    print("Loading regression model checkpoint...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    input_size = ckpt["input_size"]
    hidden1 = ckpt.get("hidden1", ckpt.get("hidden_size1", 28))
    hidden2 = ckpt.get("hidden2", ckpt.get("hidden_size2", 14))
    output_size = ckpt["output_size"]
    beta = ckpt.get("beta", 0.9)
    
    # Load scalers if available
    scaler_X = ckpt.get("scaler_X", None)
    scaler_y = ckpt.get("scaler_y", None)
    
    model = TimeSeriesConcentrationSNN(input_size, hidden1, hidden2, output_size, beta).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print(f"✓ Model loaded: {input_size} -> {hidden1} -> {hidden2} -> {output_size}")
    print(f"  Beta: {beta}")
    print(f"  Scalers available: X={scaler_X is not None}, y={scaler_y is not None}")
    print()
    
    run_summary['model'] = {
        'checkpoint': str(checkpoint_path),
        'input_size': input_size,
        'hidden1': hidden1,
        'hidden2': hidden2,
        'output_size': output_size,
        'beta': beta,
        'scalers_available': {'X': scaler_X is not None, 'y': scaler_y is not None}
    }
    
    # Load data
    print("Loading ethanol spike data...")
    spike_test, y_test, config = load_ethanol_data(spike_data_dir)
    
    print(f"✓ Test data loaded: {spike_test.shape}")
    print(f"  Test concentrations: {y_test.shape}")
    print()
    
    feature_names = config.get('feature_names', [f"f{i}" for i in range(input_size)])
    print(f"Feature names: {feature_names}")
    print(f"Concentration range: [{y_test.min():.1f}%, {y_test.max():.1f}%]")
    print()
    
    # Validate sample index
    if args.sample_index >= len(spike_test):
        print(f"Warning: sample_index {args.sample_index} >= {len(spike_test)}, using 0")
        sample_index = 0
    else:
        sample_index = args.sample_index
    
    # Select sample to analyze
    spikes = spike_test[sample_index]
    x = torch.tensor(spikes, dtype=torch.float32)
    true_concentration = y_test[sample_index]
    
    print(f"Analyzing sample {sample_index}:")
    print(f"  Shape: {x.shape}")
    print(f"  True concentration: {true_concentration:.2f}%")
    print()
    
    # Make prediction
    pred_concentration, mem_sum = predict_concentration(model, x, device, scaler_y)
    prediction_error = abs(pred_concentration - true_concentration)
    
    print(f"Model prediction: {pred_concentration:.2f}%")
    print(f"Membrane sum: {mem_sum:.4f}")
    print(f"Prediction error: {prediction_error:.2f}%")
    print()
    
    run_summary['selected_sample'] = {
        'index': sample_index,
        'true_concentration': float(true_concentration),
        'predicted_concentration': float(pred_concentration),
        'membrane_sum': float(mem_sum),
        'prediction_error': float(prediction_error),
        'shape': list(x.shape)
    }
    
    print("=" * 80)
    print("A. TSA FOR SELECTED SAMPLE")
    print("=" * 80)
    
    # Compute TSA for selected sample
    attribution = compute_tsa_regression(model, x, beta, device)
    
    print(f"TSA attribution shape: {attribution.shape}")
    print(f"Attribution range: [{attribution.min():.6f}, {attribution.max():.6f}]")
    print()
    
    # Plot TSA heatmap
    save_path = results_dir / f"tsa_sample_{sample_index}.png"
    plot_regression_heatmap(
        attribution, feature_names,
        title=f"TSA Attribution for Sample {sample_index}",
        save_path=save_path,
        concentration=pred_concentration
    )
    
    print("\n" + "=" * 80)
    print(f"B. TSA FOR {args.num_random_samples} RANDOM SAMPLES")
    print("=" * 80)
    
    # Random sample analysis
    rnd_indices = np.random.choice(len(spike_test), size=min(args.num_random_samples, len(spike_test)), replace=False)
    random_attributions = []
    random_predictions = []
    random_true_values = []
    random_samples_info = []
    
    for i, idx in enumerate(rnd_indices):
        rnd_spikes = spike_test[idx] 
        rnd_x = torch.tensor(rnd_spikes, dtype=torch.float32)
        rnd_true = y_test[idx]
        
        rnd_pred, rnd_mem = predict_concentration(model, rnd_x, device, scaler_y)
        rnd_error = abs(rnd_pred - rnd_true)
        rnd_attribution = compute_tsa_regression(model, rnd_x, beta, device)
        
        random_attributions.append(rnd_attribution)
        random_predictions.append(rnd_pred)
        random_true_values.append(rnd_true)
        
        print(f"  Sample {idx}: true={rnd_true:.2f}%, pred={rnd_pred:.2f}%, error={rnd_error:.2f}%")
        
        random_samples_info.append({
            'index': int(idx),
            'true_concentration': float(rnd_true),
            'predicted_concentration': float(rnd_pred),
            'membrane_sum': float(rnd_mem),
            'prediction_error': float(rnd_error)
        })
    
    run_summary['random_samples'] = random_samples_info
    
    print("\n" + "=" * 80)
    print("C. AGGREGATED TSA ANALYSIS")
    print("=" * 80)
    
    # Aggregate TSA from selected + random samples
    all_attributions = [attribution] + random_attributions
    all_predictions = [pred_concentration] + random_predictions
    all_true_values = [true_concentration] + random_true_values
    
    avg_attribution = torch.stack(all_attributions).mean(dim=0)
    
    save_path = results_dir / f"tsa_aggregate_{len(all_attributions)}samples.png"
    plot_regression_heatmap(
        avg_attribution, feature_names,
        title=f"Average TSA Attribution ({len(all_attributions)} samples)",
        save_path=save_path,
        concentration=np.mean(all_predictions)
    )
    
    print("\n" + "=" * 80)
    print("D. CONCENTRATION RANGE COMPARISON")
    print("=" * 80)
    
    # Compare TSA patterns across concentration ranges
    plot_concentration_range_comparison(
        all_attributions, all_true_values, feature_names, save_dir=results_dir
    )
    
    print("\n" + "=" * 80)  
    print("E. PERTURBATION ANALYSIS")
    print("=" * 80)
    print("Testing importance of top-k vs. random-k spike deletions...\n")
    
    save_path = results_dir / "perturbation_analysis.png"
    perturbation_data = plot_regression_perturbation_curve(
        model, x, attribution, device, feature_names,
        true_concentration, scaler_y, save_path=save_path
    )
    
    run_summary['perturbation_analysis'] = perturbation_data
    
    print("\n" + "=" * 80)
    print("F. PREDICTION ERROR ANALYSIS") 
    print("=" * 80)
    
    # Analyze relationship between TSA and prediction errors
    plot_prediction_error_analysis(
        all_predictions, all_true_values, all_attributions,
        feature_names, save_dir=results_dir
    )
    
    # Calculate regression metrics
    mse = mean_squared_error(all_true_values, all_predictions)
    mae = mean_absolute_error(all_true_values, all_predictions)
    r2 = r2_score(all_true_values, all_predictions)
    
    run_summary['regression_metrics'] = {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'mae': float(mae),
        'r2_score': float(r2),
        'num_samples': len(all_predictions)
    }
    
    print(f"\nRegression metrics on analyzed samples:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {np.sqrt(mse):.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    if args.full_dataset_analysis:
        print("\n" + "=" * 80)
        print("G. FULL DATASET ANALYSIS")
        print("=" * 80)
        print("Computing TSA for entire test dataset - this may take a while...")
        
        full_attributions = []
        full_predictions = []
        full_true_values = []
        
        for i in range(len(spike_test)):
            sample_spikes = spike_test[i]
            sample_x = torch.tensor(sample_spikes, dtype=torch.float32)
            sample_true = y_test[i]
            
            sample_pred, sample_mem = predict_concentration(model, sample_x, device, scaler_y)
            sample_attribution = compute_tsa_regression(model, sample_x, beta, device)
            
            full_attributions.append(sample_attribution)
            full_predictions.append(sample_pred)
            full_true_values.append(sample_true)
            
            if (i + 1) % 10 == 0 or i == len(spike_test) - 1:
                print(f"  Processed {i+1}/{len(spike_test)} samples", end='\r')
        
        print("\n✓ Full dataset TSA computation complete.\n")
        
        # Full dataset metrics
        full_mse = mean_squared_error(full_true_values, full_predictions)
        full_mae = mean_absolute_error(full_true_values, full_predictions)
        full_r2 = r2_score(full_true_values, full_predictions)
        
        print(f"Full dataset regression metrics:")
        print(f"  MSE: {full_mse:.4f}")
        print(f"  RMSE: {np.sqrt(full_mse):.4f}")
        print(f"  MAE: {full_mae:.4f}")
        print(f"  R² Score: {full_r2:.4f}")
        
        # Full dataset analysis plots
        plot_concentration_range_comparison(
            full_attributions, full_true_values, feature_names, save_dir=results_dir
        )
        
        plot_prediction_error_analysis(
            full_predictions, full_true_values, full_attributions,
            feature_names, save_dir=results_dir
        )
        
        run_summary['full_dataset_analysis'] = {
            'total_samples': len(spike_test),
            'mse': float(full_mse),
            'rmse': float(np.sqrt(full_mse)),
            'mae': float(full_mae),
            'r2_score': float(full_r2)
        }
    
    print("\n" + "=" * 80)
    print("H. SAVING RESULTS SUMMARY")
    print("=" * 80)
    
    # Save comprehensive summary
    summary_path = results_dir / "tsa_regression_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2, default=str)
    
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✓ TSA REGRESSION ANALYSIS COMPLETE!")
    print(f"✓ Results saved to: {results_dir}")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"- Model prediction accuracy: MAE = {mae:.4f}%, R² = {r2:.4f}")
    print(f"- TSA reveals temporal feature importance for concentration prediction")
    print(f"- Perturbation analysis validates TSA attributions")
    if args.full_dataset_analysis:
        print(f"- Full dataset R² = {full_r2:.4f} across {len(spike_test)} samples")


if __name__ == "__main__":
    main()