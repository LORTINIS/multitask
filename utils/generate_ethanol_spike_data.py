"""
Utility script to generate spike train data for ethanol concentration regression explainability.

This script:
1. Loads and preprocesses the ethanol time-series dataset (NO temperature/humidity)
2. Creates spike trains using direct encoding (same as training script)
3. Saves spike trains and metadata to .npy files for TSA regression explainability

Key differences from wine classification version:
- Uses ETHANOL dataset (not wine)
- CONTINUOUS concentration targets (not discrete classes)
- SEQUENCE LENGTH 1000 (not 500)
- NO temperature/humidity features (only 6 MQ sensors)
- Matches the ethanol_timeseries_snn_seq1000 model architecture

Usage:
    python generate_ethanol_spike_data.py --dataset_path ../data/wine --output_dir ../data/spike_data_ethanol
"""

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_ethanol_dataset(base_path: str) -> pd.DataFrame:
    """
    Loads the ethanol time-series dataset preserving temporal structure.
    IDENTICAL to singletask_concentration_time_series.py but WITHOUT temp/humidity
    """
    all_data = []
    
    # ONLY MQ SENSORS - NO temperature/humidity
    ts_columns = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    # Concentration mapping
    ethanol_concentration_map = {
        'C1': 1.0,    # 1%
        'C2': 2.5,    # 2.5%
        'C3': 5.0,    # 5%
        'C4': 10.0,   # 10%
        'C5': 15.0,   # 15%
        'C6': 20.0    # 20%
    }
    
    print(f"Loading ethanol data from: {base_path}")
    
    ethanol_path = os.path.join(base_path, 'Ethanol')
    
    if not os.path.exists(ethanol_path):
        raise FileNotFoundError(f"Ethanol folder not found at {ethanol_path}")
    
    print(f"Processing folder: Ethanol...")
    
    files = os.listdir(ethanol_path)
    total_files = len([f for f in files if f.endswith('.txt')])
    processed_files = 0
    
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(ethanol_path, file_name)
            
            try:
                # Load with all 8 columns initially
                all_columns = [
                    'Rel_Humidity (%)', 'Temperature (C)',
                    'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
                    'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
                ]
                
                df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=all_columns)
                
                # Keep only MQ sensor columns (drop temp/humidity)
                df_file = df_file[ts_columns].copy()
                
                df_file['Filename'] = file_name
                df_file['Time_Point'] = range(len(df_file))
                
                # Extract concentration from filename
                conc_code = file_name[3:5]  # e.g., "ET_C1_..." -> "C1"
                concentration_value = ethanol_concentration_map.get(conc_code, np.nan)
                
                if pd.isna(concentration_value):
                    print(f"  Warning: Unknown concentration code '{conc_code}' in {file_name}")
                    continue
                    
                df_file['Concentration_Value'] = concentration_value
                df_file['Concentration_Label'] = f"{concentration_value}%"
                
                # Extract repetition number
                parts = file_name.split('_')
                if len(parts) >= 3:
                    rep_part = parts[2].replace('.txt', '')
                    df_file['Repetition'] = rep_part
                else:
                    df_file['Repetition'] = 'R1'
                
                all_data.append(df_file)
                processed_files += 1
                
                if processed_files % 10 == 0:
                    print(f"  Processed {processed_files}/{total_files} files...")
                    
            except Exception as e:
                print(f"  Error processing {file_name}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No valid ethanol data files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"✓ Loaded {len(all_data)} files")
    print(f"✓ Total data points: {len(combined_df)}")
    print(f"✓ Unique concentrations: {sorted(combined_df['Concentration_Value'].unique())}")
    print(f"✓ Features: {ts_columns}")
    
    return combined_df


def preprocess_ethanol_sequences(df: pd.DataFrame, sequence_length: int = 1000, 
                                downsample_factor: int = 2, stabilization_period: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess ethanol data into time series sequences.
    MATCHES singletask_concentration_time_series.py preprocessing exactly.
    """
    
    # Only MQ sensor columns
    sensor_cols = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    print(f"\nPreprocessing sequences:")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Downsample factor: {downsample_factor}")
    print(f"  Stabilization period: {stabilization_period}")
    print(f"  Sensor features: {len(sensor_cols)}")
    
    # Remove stabilization period
    processed_files = []
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename].copy()
        if len(file_data) > stabilization_period:
            file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
            processed_files.append(file_data)
    
    df_processed = pd.concat(processed_files, ignore_index=True)
    print(f"  After removing stabilization: {len(df_processed)} data points")
    
    # Create sequences
    sequences = []
    concentrations = []
    
    for filename in df_processed['Filename'].unique():
        file_data = df_processed[df_processed['Filename'] == filename]
        
        if len(file_data) < sequence_length * downsample_factor:
            print(f"  Skipping {filename}: insufficient data ({len(file_data)} points)")
            continue
            
        # Extract sequence with downsampling
        sensor_data = file_data[sensor_cols].values
        downsampled_data = sensor_data[::downsample_factor]
        
        if len(downsampled_data) >= sequence_length:
            sequence = downsampled_data[:sequence_length]
            concentration = file_data['Concentration_Value'].iloc[0]
            
            sequences.append(sequence)
            concentrations.append(concentration)
    
    print(f"  Created {len(sequences)} sequences")
    
    # Handle missing values and normalize
    def clean_sequence(seq):
        seq_clean = seq.copy()
        for i in range(seq_clean.shape[1]):
            col = seq_clean[:, i]
            if np.any(np.isnan(col)) or np.any(np.isinf(col)):
                median_val = np.nanmedian(col[np.isfinite(col)])
                if np.isnan(median_val):
                    median_val = 0.0
                seq_clean[:, i] = np.where(np.isfinite(col), col, median_val)
        return seq_clean
    
    sequences_clean = [clean_sequence(seq) for seq in sequences]
    
    # Normalize sequences
    all_data = np.vstack(sequences_clean)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(all_data)
    
    normalized_sequences = []
    for seq in sequences_clean:
        normalized_seq = scaler_X.transform(seq)
        normalized_sequences.append(normalized_seq)
    
    # Convert to arrays
    X = np.array(normalized_sequences)  # [samples, timesteps, features]
    y = np.array(concentrations)        # [samples] - continuous concentration values
    
    print(f"  Final shapes: X={X.shape}, y={y.shape}")
    print(f"  Concentration range: [{y.min():.1f}%, {y.max():.1f}%]")
    
    return X, y, scaler_X


def encode_time_series_direct(sequences: np.ndarray) -> torch.Tensor:
    """
    Convert normalized time series to binary spikes using direct encoding.
    IDENTICAL to singletask_concentration_time_series.py encoding.
    """
    print(f"\nEncoding time series to spikes:")
    print(f"  Input shape: {sequences.shape}")
    print(f"  Encoding method: Direct (threshold = 0.5)")
    
    # Direct encoding: values > 0.5 become spikes
    spike_data = (sequences > 0.5).astype(np.float32)
    
    # Convert to tensor [samples, timesteps, features]
    spike_tensor = torch.FloatTensor(spike_data)
    
    spike_rate = spike_data.mean()
    print(f"  Spike rate: {spike_rate:.4f}")
    print(f"  Output shape: {spike_tensor.shape}")
    
    return spike_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Generate spike data for ethanol concentration regression TSA"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data/wine",
        help="Path to wine dataset directory containing Ethanol folder"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="../data/spike_data_ethanol",
        help="Output directory for spike data files"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Length of time series sequences (default: 1000)"
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=2,
        help="Downsample factor for sequences (default: 2)"
    )
    parser.add_argument(
        "--stabilization_period",
        type=int,
        default=500,
        help="Number of initial points to remove for stabilization (default: 500)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ETHANOL CONCENTRATION REGRESSION - SPIKE DATA GENERATION")
    print("=" * 80)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Features: MQ sensors only (NO temperature/humidity)")
    print()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ethanol dataset
    ethanol_df = load_ethanol_dataset(args.dataset_path)
    
    # Preprocess into sequences
    X, y, scaler_X = preprocess_ethanol_sequences(
        ethanol_df, 
        sequence_length=args.sequence_length,
        downsample_factor=args.downsample_factor,
        stabilization_period=args.stabilization_period
    )
    
    # Train-test split (same as training script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None  # No stratify for regression
    )
    
    print(f"\nTrain-test split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Train concentration range: [{y_train.min():.1f}%, {y_train.max():.1f}%]")
    print(f"  Test concentration range: [{y_test.min():.1f}%, {y_test.max():.1f}%]")
    
    # Encode to spikes
    spike_train = encode_time_series_direct(X_train)
    spike_test = encode_time_series_direct(X_test)
    
    # Save data
    print(f"\nSaving spike data to: {output_path}")
    
    np.save(output_path / 'spike_train.npy', spike_train.numpy())
    np.save(output_path / 'spike_test.npy', spike_test.numpy())
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)
    
    # Save configuration and metadata
    config = {
        'feature_names': [
            'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
            'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
        ],
        'task_type': 'regression',
        'target_type': 'concentration_percentage',
        'sequence_length': args.sequence_length,
        'downsample_factor': args.downsample_factor,
        'stabilization_period': args.stabilization_period,
        'test_size': 0.2,
        'random_state': 42,
        'input_size': X.shape[2],  # Number of features
        'output_size': 1,          # Regression output
        'spike_train_shape': list(spike_train.shape),
        'spike_test_shape': list(spike_test.shape),
        'encoding_method': 'direct',
        'concentration_range': [float(y.min()), float(y.max())],
        'unique_concentrations': sorted(np.unique(y).tolist()),
        'dataset': 'ethanol_time_series',
        'includes_temperature_humidity': False
    }
    
    np.save(output_path / 'config.npy', config)
    np.save(output_path / 'scaler.npy', scaler_X)
    
    # Save metadata
    train_metadata = {
        'filenames': [],  # Could add if needed
        'concentrations': y_train.tolist(),
        'indices': list(range(len(y_train)))
    }
    
    test_metadata = {
        'filenames': [],  # Could add if needed  
        'concentrations': y_test.tolist(),
        'indices': list(range(len(y_test)))
    }
    
    np.save(output_path / 'train_metadata.npy', train_metadata)
    np.save(output_path / 'test_metadata.npy', test_metadata)
    
    # Create README
    readme_content = f"""# Ethanol Concentration Regression Spike Data

Generated for TSA explainability analysis of ethanol concentration prediction.

## Configuration
- Dataset: Ethanol time series (NO temperature/humidity)
- Task: Regression (continuous concentration prediction)
- Sequence length: {args.sequence_length}
- Features: {len(config['feature_names'])} MQ sensors
- Encoding: Direct (threshold = 0.5)
- Train samples: {len(y_train)}
- Test samples: {len(y_test)}

## Concentration Range
- Min: {y.min():.1f}%
- Max: {y.max():.1f}%
- Unique values: {len(np.unique(y))}

## Files
- spike_train.npy: Training spike data [{spike_train.shape}]
- spike_test.npy: Test spike data [{spike_test.shape}] 
- y_train.npy: Training concentration targets [{y_train.shape}]
- y_test.npy: Test concentration targets [{y_test.shape}]
- config.npy: Configuration dictionary
- scaler.npy: MinMaxScaler for input normalization
- train_metadata.npy: Training sample metadata
- test_metadata.npy: Test sample metadata

## Usage
Load with:
```python
import numpy as np

spike_test = np.load('spike_test.npy')
y_test = np.load('y_test.npy') 
config = np.load('config.npy', allow_pickle=True).item()
```

Compatible with:
- ethanol_timeseries_snn_seq1000_beta0.9_bs16_lr0.001_ep100.pth
- tsa_singletask_concentration_regression.py
"""
    
    with open(output_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"✓ spike_train.npy: {spike_train.shape}")
    print(f"✓ spike_test.npy: {spike_test.shape}")
    print(f"✓ y_train.npy: {y_train.shape}")
    print(f"✓ y_test.npy: {y_test.shape}")
    print(f"✓ config.npy: Configuration saved")
    print(f"✓ scaler.npy: MinMaxScaler saved")
    print(f"✓ metadata files: train_metadata.npy, test_metadata.npy")
    print(f"✓ README.md: Documentation saved")
    
    print("\n" + "=" * 80)
    print("✅ ETHANOL REGRESSION SPIKE DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nDataset Summary:")
    print(f"  Task: Ethanol concentration regression")
    print(f"  Features: 6 MQ sensors (NO temperature/humidity)")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Samples: {len(y_train)} train, {len(y_test)} test")
    print(f"  Concentration range: [{y.min():.1f}%, {y.max():.1f}%]")
    print(f"  Compatible with model: ethanol_timeseries_snn_seq1000_*.pth")
    print(f"\nReady for TSA analysis with:")
    print(f"  python tsa_singletask_concentration_regression.py \\")
    print(f"    --spike_data_dir {args.output_dir}")


if __name__ == "__main__":
    main()