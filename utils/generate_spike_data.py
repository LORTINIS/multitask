"""
Utility script to generate spike train data for explainability analysis.

This script:
1. Loads and preprocesses the wine time-series dataset
2. Creates spike trains using direct encoding (same as training script)
3. Saves spike trains and metadata to .npy files for explainability scripts

Usage:
    python generate_spike_data.py --dataset_path ../../data/wine --output_dir ../../data/spike_data
"""

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_wine_dataset(base_path: str) -> pd.DataFrame:
    """
    Loads the wine time-series dataset preserving temporal structure.
    IDENTICAL to singletask_classification_time_series.py
    """
    all_data = []

    # TOGGLE TEMP HUMIDITY
    
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]

    # ts_columns = [
    #     'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
    #     'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    # ]
    
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


def remove_stabilisation(df: pd.DataFrame, stabilisation_period: int) -> pd.DataFrame:
    """
    Drop the initial stabilisation samples per file.
    IDENTICAL to singletask_classification_time_series.py
    """
    processed_files = []
    
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename].copy()
        
        if len(file_data) > stabilisation_period:
            file_data = file_data.iloc[stabilisation_period:].reset_index(drop=True)
            processed_files.append(file_data)
    
    if not processed_files:
        raise ValueError("No data remaining after stabilisation removal!")
    
    return pd.concat(processed_files, ignore_index=True)


def build_fixed_length_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int,
    downsample_factor: int
) -> Tuple[list, list, list]:
    """
    Create fixed-length sequences with optional downsampling.
    IDENTICAL to singletask_classification_time_series.py
    """
    sequences = []
    labels = []
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
        labels.append(file_data['Quality_Label'].iloc[0])
        
        metadata.append({
            'Filename': filename,
            'Quality_Label': file_data['Quality_Label'].iloc[0],
            'Brand': file_data['Brand'].iloc[0],
            'Bottle': file_data['Bottle'].iloc[0]
        })

    return sequences, labels, metadata


def clean_and_normalise_sequences(sequences: list) -> Tuple[list, MinMaxScaler]:
    """
    Replace invalid values and rescale features to [0, 1].
    IDENTICAL to singletask_classification_time_series.py
    """
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
    
    return normalized_sequences, scaler


def encode_time_series_direct(sequences: list) -> torch.Tensor:
    """
    Threshold-based encoding (values > 0.5 -> spike).
    IDENTICAL to singletask_classification_time_series.py
    
    Args:
        sequences: list of numpy arrays, each of shape [timesteps, features]
    
    Returns:
        spike_data: torch tensor of shape [timesteps, samples, features]
    """
    # Convert list to array [samples, timesteps, features]
    sequences_array = np.stack(sequences)
    
    # Convert to tensor
    sequences_tensor = torch.FloatTensor(sequences_array)
    
    # Threshold-based: values > 0.5 become spikes
    spike_data = (sequences_tensor > 0.5).float()
    
    # Reshape to [timesteps, samples, features]
    spike_data = spike_data.permute(1, 0, 2)
    
    return spike_data


def generate_and_save_spike_data(
    dataset_path: str,
    output_dir: str,
    sequence_length: int = 500,
    downsample_factor: int = 2,
    stabilisation_period: int = 500,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Main pipeline to generate spike data for explainability analysis.
    
    Args:
        dataset_path: Path to wine dataset directory
        output_dir: Directory to save generated spike data
        sequence_length: Fixed length for sequences
        downsample_factor: Downsampling factor for sequences
        stabilisation_period: Number of initial samples to remove
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    """
    print("\n" + "=" * 80)
    print("GENERATING SPIKE DATA FOR EXPLAINABILITY")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define feature columns
    sensor_cols = [
        "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
        "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
    ]
    environmental_cols = ["Rel_Humidity (%)", "Temperature (C)"]

    # TOGGLE TEMP HUMIDITY
    # feature_cols = sensor_cols 
    feature_cols = sensor_cols + environmental_cols
    
    # Load dataset
    print("\n1. Loading Wine Dataset...")
    raw_df = load_wine_dataset(dataset_path)
    
    # Remove stabilisation period
    print(f"\n2. Removing Stabilisation Period ({stabilisation_period} samples)...")
    clean_df = remove_stabilisation(raw_df, stabilisation_period)
    print(f"   Dataset shape after removal: {clean_df.shape}")
    
    # Build sequences
    print(f"\n3. Building Fixed-Length Sequences...")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Downsample factor: {downsample_factor}")
    sequences, labels, metadata = build_fixed_length_sequences(
        clean_df,
        feature_cols,
        sequence_length=sequence_length,
        downsample_factor=downsample_factor
    )
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Sequence shape: {sequences[0].shape} (timesteps, features)")
    
    # Clean and normalize
    print(f"\n4. Cleaning and Normalizing Sequences...")
    normalized_sequences, scaler = clean_and_normalise_sequences(sequences)
    all_normalized = np.vstack(normalized_sequences)
    print(f"   Normalized range: [{all_normalized.min():.3f}, {all_normalized.max():.3f}]")
    
    # Encode labels
    print(f"\n5. Encoding Labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    print(f"   Classes: {label_encoder.classes_}")
    for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        count = int(np.sum(y_encoded == encoded))
        print(f"     {label}: {count} samples (encoded as {encoded})")
    
    # Convert to array for train-test split
    print(f"\n6. Creating Fixed-Size Arrays...")
    max_length = sequence_length
    num_samples = len(normalized_sequences)
    num_features = len(feature_cols)
    
    X = np.zeros((num_samples, max_length, num_features))
    for i, seq in enumerate(normalized_sequences):
        X[i, :len(seq), :] = seq
    
    print(f"   Final data shape: {X.shape} (samples, timesteps, features)")
    
    # Train-test split
    print(f"\n7. Creating Train-Test Split...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y_encoded,
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Encode to spikes
    print(f"\n8. Encoding to Spike Trains...")
    spike_train = encode_time_series_direct([X_train[i] for i in range(len(X_train))])
    spike_test = encode_time_series_direct([X_test[i] for i in range(len(X_test))])
    print(f"   Training spike tensor shape: {spike_train.shape} (timesteps, samples, features)")
    print(f"   Test spike tensor shape: {spike_test.shape} (timesteps, samples, features)")
    
    # Calculate spike statistics
    total_train_spikes = spike_train.sum().item()
    total_test_spikes = spike_test.sum().item()
    train_spike_rate = total_train_spikes / spike_train.numel()
    test_spike_rate = total_test_spikes / spike_test.numel()
    print(f"   Training spike rate: {train_spike_rate:.4f}")
    print(f"   Test spike rate: {test_spike_rate:.4f}")
    
    # Save data
    print(f"\n9. Saving Data to {output_dir}...")
    
    # Save spike trains (permute back to [samples, timesteps, features] for easier loading)
    spike_train_permuted = spike_train.permute(1, 0, 2).numpy()
    spike_test_permuted = spike_test.permute(1, 0, 2).numpy()
    
    np.save(os.path.join(output_dir, 'spike_train.npy'), spike_train_permuted)
    np.save(os.path.join(output_dir, 'spike_test.npy'), spike_test_permuted)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metadata
    train_metadata = [metadata[i] for i in idx_train]
    test_metadata = [metadata[i] for i in idx_test]
    
    np.save(os.path.join(output_dir, 'train_metadata.npy'), train_metadata)
    np.save(os.path.join(output_dir, 'test_metadata.npy'), test_metadata)
    
    # Save configuration and scalers
    config = {
        'feature_names': feature_cols,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'sequence_length': sequence_length,
        'downsample_factor': downsample_factor,
        'stabilisation_period': stabilisation_period,
        'test_size': test_size,
        'random_state': random_state,
        'input_size': len(feature_cols),
        'output_size': len(label_encoder.classes_),
        'spike_train_shape': list(spike_train_permuted.shape),
        'spike_test_shape': list(spike_test_permuted.shape)
    }
    
    # Save as numpy dict
    np.save(os.path.join(output_dir, 'config.npy'), config)
    np.save(os.path.join(output_dir, 'scaler.npy'), scaler)
    np.save(os.path.join(output_dir, 'label_encoder.npy'), label_encoder)
    
    print(f"   ✓ spike_train.npy: {spike_train_permuted.shape}")
    print(f"   ✓ spike_test.npy: {spike_test_permuted.shape}")
    print(f"   ✓ y_train.npy: {y_train.shape}")
    print(f"   ✓ y_test.npy: {y_test.shape}")
    print(f"   ✓ train_metadata.npy")
    print(f"   ✓ test_metadata.npy")
    print(f"   ✓ config.npy")
    print(f"   ✓ scaler.npy")
    print(f"   ✓ label_encoder.npy")
    
    print("\n" + "=" * 80)
    print("SPIKE DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nData saved to: {output_dir}")
    print("\nTo use in explainability scripts:")
    print(f"  spike_data = np.load('{os.path.join(output_dir, 'spike_test.npy')}')")
    print(f"  labels = np.load('{os.path.join(output_dir, 'y_test.npy')}')")
    print(f"  config = np.load('{os.path.join(output_dir, 'config.npy')}', allow_pickle=True).item()")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate spike train data for explainability analysis"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../data/wine',
        help='Path to wine dataset directory (default: ../data/wine)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/spike_data',
        help='Directory to save spike data (default: ../data/spike_data)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=500,
        help='Fixed length for sequences (default: 500)'
    )
    parser.add_argument(
        '--downsample_factor',
        type=int,
        default=2,
        help='Downsampling factor (default: 2)'
    )
    parser.add_argument(
        '--stabilisation_period',
        type=int,
        default=500,
        help='Samples to remove from start (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    dataset_path = (script_dir / args.dataset_path).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    generate_and_save_spike_data(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        sequence_length=args.sequence_length,
        downsample_factor=args.downsample_factor,
        stabilisation_period=args.stabilisation_period
    )


if __name__ == "__main__":
    main()
