"""
Preprocessing utilities for feature engineering and normalization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def aggregate_time_series_features(df, data_type='wine'):
    """
    Aggregate time-series data into statistical features per file.
    
    Args:
        df: DataFrame with time-series data
        data_type: 'wine' or 'ethanol' to determine label extraction
        
    Returns:
        DataFrame with aggregated features
    """
    sensor_cols = [
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]
    
    environmental_cols = ['Rel_Humidity (%)', 'Temperature (C)']
    
    aggregated_data = []
    
    for filename in df['Filename'].unique():
        file_data = df[df['Filename'] == filename]
        
        # Create feature dictionary
        features = {'Filename': filename}
        
        # Get labels based on data type
        if data_type == 'wine':
            features['Quality_Label'] = file_data['Quality_Label'].iloc[0]
            features['Brand'] = file_data['Brand'].iloc[0]
            features['Bottle'] = file_data['Bottle'].iloc[0]
        else:  # ethanol
            features['Concentration_Value'] = file_data['Concentration_Value'].iloc[0]
            features['Concentration_Label'] = file_data['Concentration_Label'].iloc[0]
        
        features['Repetition'] = file_data['Repetition'].iloc[0]
        
        # Calculate statistics for each sensor
        for col in sensor_cols:
            features[f'{col}_mean'] = file_data[col].mean()
            features[f'{col}_std'] = file_data[col].std()
            features[f'{col}_min'] = file_data[col].min()
            features[f'{col}_max'] = file_data[col].max()
            features[f'{col}_median'] = file_data[col].median()
        
        # Environmental features
        for col in environmental_cols:
            features[f'{col}_mean'] = file_data[col].mean()
            features[f'{col}_std'] = file_data[col].std()
        
        aggregated_data.append(features)
    
    return pd.DataFrame(aggregated_data)


def preprocess_features(df_wine, df_ethanol):
    """
    Preprocess wine and ethanol datasets: aggregate features and normalize.
    
    Args:
        df_wine: Wine DataFrame
        df_ethanol: Ethanol DataFrame
        
    Returns:
        Dictionary containing processed data for both tasks
    """
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING AND PREPROCESSING")
    print(f"{'='*80}\n")
    
    # Aggregate features
    print("Aggregating time-series data into features...")
    df_wine_agg = aggregate_time_series_features(df_wine, data_type='wine')
    df_ethanol_agg = aggregate_time_series_features(df_ethanol, data_type='ethanol')
    
    print(f"  Wine aggregated shape: {df_wine_agg.shape}")
    print(f"  Ethanol aggregated shape: {df_ethanol_agg.shape}")
    
    # Extract features for wine (classification task)
    wine_metadata_cols = ['Filename', 'Quality_Label', 'Brand', 'Bottle', 'Repetition']
    wine_feature_cols = [col for col in df_wine_agg.columns if col not in wine_metadata_cols]
    
    X_wine = df_wine_agg[wine_feature_cols].values
    y_wine_labels = df_wine_agg['Quality_Label'].values
    
    # Encode wine quality labels
    label_encoder = LabelEncoder()
    y_wine = label_encoder.fit_transform(y_wine_labels)
    
    print(f"\nWine Classification Task:")
    print(f"  Features: {len(wine_feature_cols)}")
    print(f"  Samples: {len(X_wine)}")
    print(f"  Classes: {label_encoder.classes_}")
    for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        count = np.sum(y_wine == encoded)
        print(f"    {label}: {count} samples (encoded as {encoded})")
    
    # Extract features for ethanol (regression task)
    ethanol_metadata_cols = ['Filename', 'Concentration_Value', 'Concentration_Label', 'Repetition']
    ethanol_feature_cols = [col for col in df_ethanol_agg.columns if col not in ethanol_metadata_cols]
    
    X_ethanol = df_ethanol_agg[ethanol_feature_cols].values
    y_ethanol = df_ethanol_agg['Concentration_Value'].values
    
    print(f"\nEthanol Regression Task:")
    print(f"  Features: {len(ethanol_feature_cols)}")
    print(f"  Samples: {len(X_ethanol)}")
    print(f"  Concentration range: [{y_ethanol.min():.1f}%, {y_ethanol.max():.1f}%]")
    unique_concentrations = np.unique(y_ethanol)
    for conc in sorted(unique_concentrations):
        count = np.sum(y_ethanol == conc)
        print(f"    {conc}%: {count} samples")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    for X, name in [(X_wine, 'Wine'), (X_ethanol, 'Ethanol')]:
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.isnan(col_data).any() or np.isinf(col_data).any():
                col_mean = np.nanmean(col_data[np.isfinite(col_data)])
                X[:, col_idx] = np.where(np.isnan(col_data) | np.isinf(col_data), col_mean, col_data)
    
    # Normalize features to [0, 1] for spike encoding
    print(f"Normalizing features to [0, 1]...")
    scaler_X_wine = MinMaxScaler(feature_range=(0, 1))
    X_wine_normalized = scaler_X_wine.fit_transform(X_wine)
    
    scaler_X_ethanol = MinMaxScaler(feature_range=(0, 1))
    X_ethanol_normalized = scaler_X_ethanol.fit_transform(X_ethanol)
    
    # Normalize ethanol concentrations for training
    scaler_y_ethanol = MinMaxScaler(feature_range=(0, 1))
    y_ethanol_normalized = scaler_y_ethanol.fit_transform(y_ethanol.reshape(-1, 1)).flatten()
    
    print(f"\n  Wine features normalized: [{X_wine_normalized.min():.3f}, {X_wine_normalized.max():.3f}]")
    print(f"  Ethanol features normalized: [{X_ethanol_normalized.min():.3f}, {X_ethanol_normalized.max():.3f}]")
    print(f"  Ethanol target normalized: [{y_ethanol_normalized.min():.3f}, {y_ethanol_normalized.max():.3f}]")
    
    return {
        'wine': {
            'X': X_wine_normalized,
            'y': y_wine,
            'y_labels': y_wine_labels,
            'label_encoder': label_encoder,
            'scaler_X': scaler_X_wine,
            'feature_cols': wine_feature_cols
        },
        'ethanol': {
            'X': X_ethanol_normalized,
            'y': y_ethanol_normalized,
            'y_original': y_ethanol,
            'scaler_X': scaler_X_ethanol,
            'scaler_y': scaler_y_ethanol,
            'feature_cols': ethanol_feature_cols
        }
    }


def create_train_test_split(preprocessed_data, test_size=0.2, random_state=42):
    """
    Create train-test splits for both tasks.
    
    Args:
        preprocessed_data: Dictionary from preprocess_features()
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
    
    # Wine classification split (stratified)
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine_data['X'], wine_data['y'],
        test_size=test_size,
        random_state=random_state,
        stratify=wine_data['y']
    )
    
    print(f"Wine Classification Split:")
    print(f"  Train: {len(X_train_wine)} samples")
    print(f"  Test: {len(X_test_wine)} samples")
    
    # Ethanol regression split
    X_train_ethanol, X_test_ethanol, y_train_ethanol, y_test_ethanol = train_test_split(
        ethanol_data['X'], ethanol_data['y'],
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nEthanol Regression Split:")
    print(f"  Train: {len(X_train_ethanol)} samples")
    print(f"  Test: {len(X_test_ethanol)} samples")
    
    return {
        'wine': {
            'X_train': X_train_wine,
            'X_test': X_test_wine,
            'y_train': y_train_wine,
            'y_test': y_test_wine,
            'label_encoder': wine_data['label_encoder']
        },
        'ethanol': {
            'X_train': X_train_ethanol,
            'X_test': X_test_ethanol,
            'y_train': y_train_ethanol,
            'y_test': y_test_ethanol,
            'scaler_y': ethanol_data['scaler_y']
        }
    }
