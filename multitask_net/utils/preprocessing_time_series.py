"""
Preprocessing utilities for time series feature engineering and normalization.
This version preserves temporal structure for direct spike encoding.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def prepare_time_series_sequences(df, data_type='wine', sequence_length=None, 
                                   downsample_factor=1):
    """
    Prepare time-series data for direct spike encoding by organizing into sequences.
    
    Args:
        df: DataFrame with time-series data
        data_type: 'wine' or 'ethanol' to determine label extraction
        sequence_length: Fixed length for all sequences (if None, use variable length)
        downsample_factor: Factor to downsample sequences (1 = no downsampling, 2 = every 2nd point, etc.)
        
    Returns:
        Dictionary with sequences, labels, and metadata
    """
    sensor_cols = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    environmental_cols = ['Rel_Humidity (%)', 'Temperature (C)']
    feature_cols = sensor_cols + environmental_cols
    
    sequences = []
    labels = []
    metadata = []
    
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename].copy()
        
        # Extract time series features
        sequence = file_data[feature_cols].values
        
        # Downsample if specified
        if downsample_factor > 1:
            sequence = sequence[::downsample_factor]
        
        # Pad or truncate to fixed length if specified
        if sequence_length is not None:
            if len(sequence) < sequence_length:
                # Pad with last value
                padding = np.repeat(sequence[-1:], sequence_length - len(sequence), axis=0)
                sequence = np.vstack([sequence, padding])
            elif len(sequence) > sequence_length:
                # Truncate
                sequence = sequence[:sequence_length]
        
        sequences.append(sequence)
        
        # Extract labels based on data type
        meta = {'Filename': filename}
        
        if data_type == 'wine':
            label = file_data['Quality_Label'].iloc[0]
            labels.append(label)
            meta['Quality_Label'] = label
            meta['Brand'] = file_data['Brand'].iloc[0]
            meta['Bottle'] = file_data['Bottle'].iloc[0]
        else:  # ethanol
            label = file_data['Concentration_Value'].iloc[0]
            labels.append(label)
            meta['Concentration_Value'] = label
            meta['Concentration_Label'] = file_data['Concentration_Label'].iloc[0]
        
        meta['Repetition'] = file_data['Repetition'].iloc[0]
        meta['Original_Length'] = len(file_data)
        meta['Sequence_Length'] = len(sequence)
        
        metadata.append(meta)
    
    return {
        'sequences': sequences,
        'labels': labels,
        'metadata': metadata,
        'feature_names': feature_cols
    }


def normalize_sequences(sequences, scaler=None, fit=True):
    """
    Normalize time series sequences feature-wise.
    
    Args:
        sequences: List of numpy arrays with shape [time_steps, features]
        scaler: Optional pre-fitted scaler
        fit: Whether to fit the scaler (True for training data)
        
    Returns:
        Normalized sequences and fitted scaler
    """
    # Stack all sequences to fit scaler on entire dataset
    all_data = np.vstack(sequences)
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    
    if fit:
        scaler.fit(all_data)
    
    # Normalize each sequence individually
    normalized_sequences = []
    for seq in sequences:
        normalized_seq = scaler.transform(seq)
        normalized_sequences.append(normalized_seq)
    
    return normalized_sequences, scaler


def preprocess_time_series_features(df_wine, df_ethanol, sequence_length=None, 
                                     downsample_factor=1):
    """
    Preprocess wine and ethanol datasets: prepare sequences and normalize.
    
    Args:
        df_wine: Wine DataFrame
        df_ethanol: Ethanol DataFrame
        sequence_length: Fixed length for all sequences (None = variable length)
        downsample_factor: Downsampling factor for sequences
        
    Returns:
        Dictionary containing processed data for both tasks
    """
    print(f"\n{'='*80}")
    print("TIME SERIES FEATURE ENGINEERING AND PREPROCESSING")
    print(f"{'='*80}\n")
    
    # Prepare time series sequences
    print("Preparing time series sequences...")
    wine_data = prepare_time_series_sequences(
        df_wine, data_type='wine', 
        sequence_length=sequence_length,
        downsample_factor=downsample_factor
    )
    ethanol_data = prepare_time_series_sequences(
        df_ethanol, data_type='ethanol',
        sequence_length=sequence_length,
        downsample_factor=downsample_factor
    )
    
    print(f"  Wine sequences: {len(wine_data['sequences'])} samples")
    print(f"  Ethanol sequences: {len(ethanol_data['sequences'])} samples")
    
    if sequence_length:
        print(f"  Fixed sequence length: {sequence_length} time steps")
    else:
        wine_lengths = [len(seq) for seq in wine_data['sequences']]
        ethanol_lengths = [len(seq) for seq in ethanol_data['sequences']]
        print(f"  Wine sequence length range: [{min(wine_lengths)}, {max(wine_lengths)}]")
        print(f"  Ethanol sequence length range: [{min(ethanol_lengths)}, {max(ethanol_lengths)}]")
    
    print(f"  Number of features: {len(wine_data['feature_names'])}")
    print(f"  Features: {wine_data['feature_names']}")
    
    # Handle missing/infinite values
    print(f"\nHandling missing values in sequences...")
    
    def clean_sequence(seq):
        """Replace NaN and inf values with column mean."""
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
    
    wine_data['sequences'] = [clean_sequence(seq) for seq in wine_data['sequences']]
    ethanol_data['sequences'] = [clean_sequence(seq) for seq in ethanol_data['sequences']]
    
    # Normalize sequences
    print(f"Normalizing sequences to [0, 1]...")
    wine_sequences_norm, scaler_wine = normalize_sequences(
        wine_data['sequences'], fit=True
    )
    ethanol_sequences_norm, scaler_ethanol = normalize_sequences(
        ethanol_data['sequences'], fit=True
    )
    
    # Check normalization
    all_wine_vals = np.vstack(wine_sequences_norm)
    all_ethanol_vals = np.vstack(ethanol_sequences_norm)
    print(f"  Wine sequences normalized: [{all_wine_vals.min():.3f}, {all_wine_vals.max():.3f}]")
    print(f"  Ethanol sequences normalized: [{all_ethanol_vals.min():.3f}, {all_ethanol_vals.max():.3f}]")
    
    # Process wine labels
    wine_labels = np.array(wine_data['labels'])
    label_encoder = LabelEncoder()
    wine_labels_encoded = label_encoder.fit_transform(wine_labels)
    
    print(f"\nWine Classification Task:")
    print(f"  Samples: {len(wine_sequences_norm)}")
    print(f"  Classes: {label_encoder.classes_}")
    for label, encoded in zip(label_encoder.classes_, 
                              label_encoder.transform(label_encoder.classes_)):
        count = np.sum(wine_labels_encoded == encoded)
        print(f"    {label}: {count} samples (encoded as {encoded})")
    
    # Process ethanol labels
    ethanol_labels = np.array(ethanol_data['labels'])
    
    # Normalize ethanol concentrations for training
    scaler_y_ethanol = MinMaxScaler(feature_range=(0, 1))
    ethanol_labels_normalized = scaler_y_ethanol.fit_transform(
        ethanol_labels.reshape(-1, 1)
    ).flatten()
    
    print(f"\nEthanol Regression Task:")
    print(f"  Samples: {len(ethanol_sequences_norm)}")
    print(f"  Concentration range: [{ethanol_labels.min():.1f}%, {ethanol_labels.max():.1f}%]")
    unique_concentrations = np.unique(ethanol_labels)
    for conc in sorted(unique_concentrations):
        count = np.sum(ethanol_labels == conc)
        print(f"    {conc}%: {count} samples")
    
    return {
        'wine': {
            'sequences': wine_sequences_norm,
            'y': wine_labels_encoded,
            'y_labels': wine_labels,
            'label_encoder': label_encoder,
            'scaler_X': scaler_wine,
            'metadata': wine_data['metadata'],
            'feature_names': wine_data['feature_names']
        },
        'ethanol': {
            'sequences': ethanol_sequences_norm,
            'y': ethanol_labels_normalized,
            'y_original': ethanol_labels,
            'scaler_X': scaler_ethanol,
            'scaler_y': scaler_y_ethanol,
            'metadata': ethanol_data['metadata'],
            'feature_names': ethanol_data['feature_names']
        }
    }


def pad_sequences_to_max_length(sequences):
    """
    Pad variable-length sequences to the maximum length.
    
    Args:
        sequences: List of numpy arrays with shape [time_steps, features]
        
    Returns:
        Padded numpy array with shape [num_samples, max_time_steps, features]
        and array of original lengths
    """
    max_length = max(len(seq) for seq in sequences)
    num_features = sequences[0].shape[1]
    num_samples = len(sequences)
    
    padded = np.zeros((num_samples, max_length, num_features))
    lengths = np.zeros(num_samples, dtype=int)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len, :] = seq
        # Pad remaining with last value
        if seq_len < max_length:
            padded[i, seq_len:, :] = seq[-1]
        lengths[i] = seq_len
    
    return padded, lengths


def create_train_test_split_sequences(preprocessed_data, test_size=0.2, random_state=42):
    """
    Create train-test splits for time series data.
    
    Args:
        preprocessed_data: Dictionary from preprocess_time_series_features()
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with train/test splits for both tasks
    """
    print(f"\n{'='*80}")
    print("CREATING TRAIN-TEST SPLITS")
    print(f"{'='*80}\n")
    
    wine_data = preprocessed_data['wine']
    ethanol_data = preprocessed_data['ethanol']
    
    # Convert sequences to arrays
    wine_sequences_array, wine_lengths = pad_sequences_to_max_length(wine_data['sequences'])
    ethanol_sequences_array, ethanol_lengths = pad_sequences_to_max_length(ethanol_data['sequences'])
    
    print(f"Padded sequence shapes:")
    print(f"  Wine: {wine_sequences_array.shape} (samples, time_steps, features)")
    print(f"  Ethanol: {ethanol_sequences_array.shape} (samples, time_steps, features)")
    
    # Wine classification split (stratified)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine, \
    train_idx_wine, test_idx_wine = train_test_split(
        wine_sequences_array, wine_data['y'],
        np.arange(len(wine_sequences_array)),
        test_size=test_size,
        random_state=random_state,
        stratify=wine_data['y']
    )
    
    train_lengths_wine = wine_lengths[train_idx_wine]
    test_lengths_wine = wine_lengths[test_idx_wine]
    
    print(f"\nWine Classification Split:")
    print(f"  Train: {len(X_train_wine)} samples")
    print(f"  Test: {len(X_test_wine)} samples")
    
    # Ethanol regression split
    X_train_ethanol, X_test_ethanol, y_train_ethanol, y_test_ethanol, \
    train_idx_ethanol, test_idx_ethanol = train_test_split(
        ethanol_sequences_array, ethanol_data['y'],
        np.arange(len(ethanol_sequences_array)),
        test_size=test_size,
        random_state=random_state
    )
    
    train_lengths_ethanol = ethanol_lengths[train_idx_ethanol]
    test_lengths_ethanol = ethanol_lengths[test_idx_ethanol]
    
    print(f"\nEthanol Regression Split:")
    print(f"  Train: {len(X_train_ethanol)} samples")
    print(f"  Test: {len(X_test_ethanol)} samples")
    
    return {
        'wine': {
            'X_train': X_train_wine,
            'X_test': X_test_wine,
            'y_train': y_train_wine,
            'y_test': y_test_wine,
            'train_lengths': train_lengths_wine,
            'test_lengths': test_lengths_wine,
            'label_encoder': wine_data['label_encoder'],
            'feature_names': wine_data['feature_names']
        },
        'ethanol': {
            'X_train': X_train_ethanol,
            'X_test': X_test_ethanol,
            'y_train': y_train_ethanol,
            'y_test': y_test_ethanol,
            'train_lengths': train_lengths_ethanol,
            'test_lengths': test_lengths_ethanol,
            'scaler_y': ethanol_data['scaler_y'],
            'feature_names': ethanol_data['feature_names']
        }
    }


# Example usage:
"""
# In your training script, replace the preprocessing call:

# OLD:
# preprocessed_data = preprocess_features(wine_df, ethanol_df)

# NEW:
preprocessed_data = preprocess_time_series_features(
    wine_df, 
    ethanol_df, 
    sequence_length=500,  # Fixed length, or None for variable
    downsample_factor=2    # Take every 2nd point, or 1 for no downsampling
)

splits = create_train_test_split_sequences(
    preprocessed_data, 
    test_size=0.2, 
    random_state=42
)

# Now X_train and X_test are 3D arrays: [samples, time_steps, features]
# You can directly encode these as spike trains

# For spike encoding:
# Input shape: [samples, time_steps, features]
# You can use each time step as input to the SNN, or
# Use rate encoding on the entire sequence
"""