"""
Feature Correlation Analysis for Spike Data
============================================

This utility analyzes correlations between features in spike data to identify:
    - Feature-feature correlations (Pearson, Spearman)
    - Feature-label correlations
    - Partial correlations (controlling for temperature/humidity)
    - Mutual information between features and labels

Usage:
    python feature_correlation.py \
        --spike_data_dir ../data/spike_data \
        --output_dir ../results/correlation_analysis

This helps identify spurious correlations that might affect explainability results.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif


def load_spike_data(spike_data_dir):
    """
    Load spike data and metadata from directory.
    
    Returns:
        dict with keys: spike_train, spike_test, y_train, y_test, 
                       config, feature_names, class_names
    """
    spike_data_dir = Path(spike_data_dir)
    
    spike_train = np.load(spike_data_dir / 'spike_train.npy')
    spike_test = np.load(spike_data_dir / 'spike_test.npy')
    y_train = np.load(spike_data_dir / 'y_train.npy')
    y_test = np.load(spike_data_dir / 'y_test.npy')
    config = np.load(spike_data_dir / 'config.npy', allow_pickle=True).item()
    
    feature_names = config.get('feature_names', [f"f{i}" for i in range(spike_train.shape[2])])
    class_names = config.get('label_encoder_classes', [f"c{i}" for i in range(len(np.unique(y_train)))])
    
    return {
        'spike_train': spike_train,
        'spike_test': spike_test,
        'y_train': y_train,
        'y_test': y_test,
        'config': config,
        'feature_names': feature_names,
        'class_names': class_names
    }


def compute_spike_rates(spike_data):
    """
    Compute spike rate per feature as summary statistic.
    
    Args:
        spike_data: Array of shape [samples, timesteps, features]
    
    Returns:
        Array of shape [samples, features] with spike rates
    """
    # Sum over time, normalize by timesteps
    spike_rates = spike_data.sum(axis=1) / spike_data.shape[1]
    return spike_rates


def compute_pearson_correlation(data, feature_names):
    """
    Compute Pearson correlation matrix.
    
    Args:
        data: Array of shape [samples, features]
        feature_names: List of feature names
    
    Returns:
        DataFrame with correlation matrix
    """
    corr_matrix = np.corrcoef(data, rowvar=False)
    df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    return df


def compute_spearman_correlation(data, feature_names):
    """
    Compute Spearman rank correlation matrix.
    
    Args:
        data: Array of shape [samples, features]
        feature_names: List of feature names
    
    Returns:
        DataFrame with correlation matrix
    """
    corr_matrix, _ = spearmanr(data)
    df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    return df


def compute_partial_correlation(data, feature_idx, control_indices, feature_names):
    """
    Compute partial correlation controlling for specific features.
    
    Args:
        data: Array of shape [samples, features]
        feature_idx: Index of feature to analyze
        control_indices: List of indices to control for
        feature_names: List of feature names
    
    Returns:
        Series with partial correlations
    """
    n_features = data.shape[1]
    partial_corrs = []
    
    for i in range(n_features):
        if i == feature_idx:
            partial_corrs.append(1.0)
            continue
        
        # Compute residuals after controlling for control variables
        from sklearn.linear_model import LinearRegression
        
        if len(control_indices) > 0:
            X_control = data[:, control_indices]
            
            # Residuals for feature_idx
            reg1 = LinearRegression()
            reg1.fit(X_control, data[:, feature_idx])
            resid1 = data[:, feature_idx] - reg1.predict(X_control)
            
            # Residuals for feature i
            reg2 = LinearRegression()
            reg2.fit(X_control, data[:, i])
            resid2 = data[:, i] - reg2.predict(X_control)
            
            # Correlation of residuals
            partial_corr = np.corrcoef(resid1, resid2)[0, 1]
        else:
            # No control variables, return regular correlation
            partial_corr = np.corrcoef(data[:, feature_idx], data[:, i])[0, 1]
        
        partial_corrs.append(partial_corr)
    
    return pd.Series(partial_corrs, index=feature_names, name=f"{feature_names[feature_idx]}_partial")


def compute_feature_label_correlation(data, labels, feature_names, class_names):
    """
    Compute correlation between features and class labels.
    
    Args:
        data: Array of shape [samples, features]
        labels: Array of shape [samples]
        feature_names: List of feature names
        class_names: List of class names
    
    Returns:
        DataFrame with correlations for each class (one-vs-rest)
    """
    n_classes = len(np.unique(labels))
    n_features = data.shape[1]
    
    correlations = {}
    
    for class_idx in range(n_classes):
        binary_labels = (labels == class_idx).astype(float)
        class_corrs = []
        
        for feat_idx in range(n_features):
            corr = np.corrcoef(data[:, feat_idx], binary_labels)[0, 1]
            class_corrs.append(corr)
        
        correlations[class_names[class_idx]] = class_corrs
    
    df = pd.DataFrame(correlations, index=feature_names)
    return df


def compute_mutual_information(data, labels, feature_names):
    """
    Compute mutual information between features and labels.
    
    Args:
        data: Array of shape [samples, features]
        labels: Array of shape [samples]
        feature_names: List of feature names
    
    Returns:
        Series with mutual information scores
    """
    mi_scores = mutual_info_classif(data, labels, random_state=42)
    return pd.Series(mi_scores, index=feature_names, name='Mutual_Information')


def plot_correlation_heatmap(corr_df, title, save_path, vmin=-1, vmax=1, cmap='coolwarm'):
    """
    Plot correlation heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap=cmap, center=0,
                vmin=vmin, vmax=vmax, square=True, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_label_correlation(corr_df, title, save_path):
    """
    Plot feature-label correlation heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-0.5, vmax=0.5, linewidths=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=11)
    plt.ylabel('Classes (One-vs-Rest)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def analyze_correlations(spike_data_dir, output_dir, dataset_name="spike_data"):
    """
    Run full correlation analysis on spike data.
    
    Args:
        spike_data_dir: Path to directory containing spike data
        output_dir: Path to output directory
        dataset_name: Name to use for labeling outputs
    """
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    
    # Load data
    print("Loading spike data...")
    data_dict = load_spike_data(spike_data_dir)
    
    spike_train = data_dict['spike_train']
    spike_test = data_dict['spike_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    class_names = data_dict['class_names']
    
    print(f"  Train: {spike_train.shape}, Test: {spike_test.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Classes: {class_names}")
    
    # Combine train and test for analysis
    all_spikes = np.concatenate([spike_train, spike_test], axis=0)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    
    # Compute spike rates as features
    print("\nComputing spike rates...")
    spike_rates = compute_spike_rates(all_spikes)
    print(f"  Spike rates shape: {spike_rates.shape}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(all_spikes),
        'n_features': len(feature_names),
        'n_classes': len(class_names),
        'feature_names': feature_names,
        'class_names': class_names
    }
    
    # 1. Pearson correlation
    print("\n1. Computing Pearson correlations...")
    pearson_corr = compute_pearson_correlation(spike_rates, feature_names)
    pearson_path = output_dir / f"{dataset_name}_pearson_correlation.csv"
    pearson_corr.to_csv(pearson_path)
    print(f"  Saved: {pearson_path}")
    
    plot_correlation_heatmap(
        pearson_corr,
        f"Pearson Correlation - {dataset_name}",
        output_dir / f"{dataset_name}_pearson_heatmap.png"
    )
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = pearson_corr.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': float(corr_val)
                })
    
    analysis_results['high_pearson_correlations'] = high_corr_pairs
    print(f"  Found {len(high_corr_pairs)} high correlations (|r| > 0.7)")
    
    # 2. Spearman correlation
    print("\n2. Computing Spearman correlations...")
    spearman_corr = compute_spearman_correlation(spike_rates, feature_names)
    spearman_path = output_dir / f"{dataset_name}_spearman_correlation.csv"
    spearman_corr.to_csv(spearman_path)
    print(f"  Saved: {spearman_path}")
    
    plot_correlation_heatmap(
        spearman_corr,
        f"Spearman Correlation - {dataset_name}",
        output_dir / f"{dataset_name}_spearman_heatmap.png"
    )
    
    # 3. Feature-label correlation
    print("\n3. Computing feature-label correlations...")
    feature_label_corr = compute_feature_label_correlation(
        spike_rates, all_labels, feature_names, class_names
    )
    fl_corr_path = output_dir / f"{dataset_name}_feature_label_correlation.csv"
    feature_label_corr.to_csv(fl_corr_path)
    print(f"  Saved: {fl_corr_path}")
    
    plot_feature_label_correlation(
        feature_label_corr,
        f"Feature-Label Correlation - {dataset_name}",
        output_dir / f"{dataset_name}_feature_label_heatmap.png"
    )
    
    # Find features highly correlated with labels
    high_label_corrs = []
    for feat_idx, feat_name in enumerate(feature_names):
        for class_idx, class_name in enumerate(class_names):
            corr_val = feature_label_corr.loc[feat_name, class_name]
            if abs(corr_val) > 0.3:
                high_label_corrs.append({
                    'feature': feat_name,
                    'class': class_name,
                    'correlation': float(corr_val)
                })
    
    analysis_results['high_feature_label_correlations'] = high_label_corrs
    print(f"  Found {len(high_label_corrs)} high feature-label correlations (|r| > 0.3)")
    
    # 4. Mutual information
    print("\n4. Computing mutual information...")
    mi_scores = compute_mutual_information(spike_rates, all_labels, feature_names)
    mi_path = output_dir / f"{dataset_name}_mutual_information.csv"
    mi_scores.to_csv(mi_path)
    print(f"  Saved: {mi_path}")
    
    # Plot MI scores
    plt.figure(figsize=(10, 6))
    mi_scores.sort_values(ascending=False).plot(kind='bar')
    plt.title(f"Mutual Information with Labels - {dataset_name}", fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=11)
    plt.ylabel('Mutual Information', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    mi_plot_path = output_dir / f"{dataset_name}_mutual_information.png"
    plt.savefig(mi_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {mi_plot_path}")
    plt.close()
    
    analysis_results['mutual_information'] = {
        feat: float(score) for feat, score in mi_scores.items()
    }
    
    # 5. Partial correlations (if temperature/humidity present)
    if len(feature_names) >= 10:  # Assume last 2 might be T/H
        print("\n5. Computing partial correlations (controlling for potential T/H features)...")
        
        # Try to identify T/H features by name
        th_indices = []
        for idx, name in enumerate(feature_names):
            if 'temp' in name.lower() or 'humid' in name.lower() or 'th' in name.lower():
                th_indices.append(idx)
        
        if len(th_indices) >= 2:
            print(f"  Detected T/H features at indices {th_indices}: {[feature_names[i] for i in th_indices]}")
            
            # Compute partial correlations for sensor features
            sensor_indices = [i for i in range(len(feature_names)) if i not in th_indices]
            
            partial_corr_matrix = np.eye(len(feature_names))
            for i in sensor_indices:
                partial_corrs = compute_partial_correlation(
                    spike_rates, i, th_indices, feature_names
                )
                partial_corr_matrix[i, :] = partial_corrs.values
                partial_corr_matrix[:, i] = partial_corrs.values
            
            partial_df = pd.DataFrame(
                partial_corr_matrix, 
                index=feature_names, 
                columns=feature_names
            )
            
            partial_path = output_dir / f"{dataset_name}_partial_correlation.csv"
            partial_df.to_csv(partial_path)
            print(f"  Saved: {partial_path}")
            
            plot_correlation_heatmap(
                partial_df,
                f"Partial Correlation (controlling for T/H) - {dataset_name}",
                output_dir / f"{dataset_name}_partial_heatmap.png"
            )
        else:
            print("  No T/H features detected by name; skipping partial correlation")
    
    # Save summary
    summary_path = output_dir / f"{dataset_name}_correlation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\n✓ Summary saved: {summary_path}")
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature correlations in spike data"
    )
    parser.add_argument(
        "--spike_data_dir",
        type=str,
        default="../data/spike_data",
        help="Directory containing spike data (default: ../data/spike_data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/correlation_analysis",
        help="Output directory for results (default: ../results/correlation_analysis)"
    )
    parser.add_argument(
        "--compare_datasets",
        action="store_true",
        help="If set, also analyze spike_data_wo_th and compare"
    )
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%H%M_%d%m%Y")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}\n")
    
    # Analyze primary dataset
    spike_data_dir = Path(args.spike_data_dir)
    if spike_data_dir.exists():
        results1 = analyze_correlations(spike_data_dir, output_dir, "with_th")
    else:
        print(f"ERROR: Spike data directory not found: {spike_data_dir}")
        return
    
    # Optionally compare with wo_th dataset
    if args.compare_datasets:
        spike_data_dir_wo_th = spike_data_dir.parent / "spike_data_wo_th"
        if spike_data_dir_wo_th.exists():
            results2 = analyze_correlations(spike_data_dir_wo_th, output_dir, "without_th")
            
            # Create comparison summary
            print(f"\n{'='*80}")
            print("COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(f"\nWith T/H: {results1['n_features']} features")
            print(f"Without T/H: {results2['n_features']} features")
            print(f"\nHigh correlations (|r| > 0.7):")
            print(f"  With T/H: {len(results1['high_pearson_correlations'])}")
            print(f"  Without T/H: {len(results2['high_pearson_correlations'])}")
        else:
            print(f"\nNote: spike_data_wo_th not found at {spike_data_dir_wo_th}")
    
    print(f"\n{'='*80}")
    print("✓ CORRELATION ANALYSIS COMPLETE!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
