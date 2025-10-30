"""
Data loading utilities for wine and ethanol datasets.
"""

import os
import numpy as np
import pandas as pd


def load_wine_dataset(base_path):
    """
    Loads the wine time-series dataset from the specified directory.
    
    Args:
        base_path: Path to the base directory containing wine data folders
        
    Returns:
        DataFrame with wine data and quality labels
    """
    all_data = []

    # Define the expected columns
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]

    print(f"Loading wine dataset from: {base_path}")

    for root, _, files in os.walk(base_path):
        folder_name = os.path.basename(root)

        # Only process files inside wine data directories (case-insensitive)
        if folder_name.lower() in ['lq_wines', 'hq_wines', 'aq_wines']:
            print(f"  Processing folder: {folder_name}...")

            for file_name in files:
                if file_name.endswith('.txt'):
                    file_path = os.path.join(root, file_name)

                    try:
                        # Read the time-series data
                        df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=ts_columns)
                        df_file['Filename'] = file_name
                        df_file['Time_Point'] = range(len(df_file))

                        # Extract Labels
                        df_file['Data_Type'] = 'Wine'
                        df_file['Quality_Label'] = folder_name.split('_')[0][:2].upper()

                        # Using character positions from file description
                        df_file['Brand'] = file_name[3:9]
                        df_file['Bottle'] = file_name[10:13]

                        # Find repetition part (R01)
                        rep_start_index = file_name.rfind('_R') + 1
                        df_file['Repetition'] = file_name[rep_start_index:rep_start_index + 3]

                        all_data.append(df_file)

                    except Exception as e:
                        print(f"    Error processing file {file_name}: {e}")
                        continue

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)

    # Final column reorder
    label_cols = ['Data_Type', 'Quality_Label', 'Brand', 'Bottle', 'Repetition', 'Filename', 'Time_Point']
    final_df = final_df[label_cols + ts_columns]
    
    print(f"  Wine data loaded: {len(final_df)} rows, {len(final_df['Filename'].unique())} files")
    return final_df


def load_ethanol_dataset(base_path):
    """
    Loads the ethanol concentration time-series dataset.
    
    Args:
        base_path: Path to the base directory containing ethanol data
        
    Returns:
        DataFrame with ethanol data and concentration labels
    """
    all_data = []

    # Define the expected columns
    ts_columns = [
        'Rel_Humidity (%)', 'Temperature (C)',
        'MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', 'MQ-6_R1 (kOhm)',
        'MQ-3_R2 (kOhm)', 'MQ-4_R2 (kOhm)', 'MQ-6_R2 (kOhm)'
    ]

    # Define concentration mapping for Ethanol files
    ethanol_concentration_map = {
        'C1': 1.0,    # 1%
        'C2': 2.5,    # 2.5%
        'C3': 5.0,    # 5%
        'C4': 10.0,   # 10%
        'C5': 15.0,   # 15%
        'C6': 20.0    # 20%
    }

    print(f"Loading ethanol dataset from: {base_path}")

    # Look for Ethanol folder
    ethanol_path = os.path.join(base_path, 'Ethanol')
    
    if not os.path.exists(ethanol_path):
        print(f"  ERROR: Ethanol folder not found at {ethanol_path}")
        return pd.DataFrame()
    
    print(f"  Processing folder: Ethanol...")
    
    files = os.listdir(ethanol_path)
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(ethanol_path, file_name)

            try:
                # Read the time-series data
                df_file = pd.read_csv(file_path, sep=r'\s+', header=None, names=ts_columns)
                df_file['Filename'] = file_name
                df_file['Time_Point'] = range(len(df_file))

                # Extract Concentration Label
                conc_code = file_name[3:5]
                df_file['Data_Type'] = 'Ethanol'
                df_file['Concentration_Value'] = ethanol_concentration_map.get(conc_code, np.nan)
                df_file['Concentration_Label'] = f"{ethanol_concentration_map.get(conc_code, 'Unknown')}%"

                # Repetition part (R01, R02, etc.)
                rep_start_index = file_name.rfind('R')
                df_file['Repetition'] = file_name[rep_start_index:rep_start_index + 3] if rep_start_index != -1 else 'Unknown'

                all_data.append(df_file)

            except Exception as e:
                print(f"    Error processing file {file_name}: {e}")
                continue

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)

    # Final column reorder
    label_cols = ['Data_Type', 'Concentration_Value', 'Concentration_Label', 'Repetition', 'Filename', 'Time_Point']
    final_df = final_df[label_cols + ts_columns]
    
    print(f"  Ethanol data loaded: {len(final_df)} rows, {len(final_df['Filename'].unique())} files")
    print(f"  Unique concentrations: {sorted(final_df['Concentration_Value'].dropna().unique())}")
    return final_df


def load_combined_dataset(base_path, stabilization_period=500):
    """
    Loads both wine and ethanol datasets and removes stabilization period.
    
    Args:
        base_path: Path to the base directory
        stabilization_period: Number of initial time points to remove
        
    Returns:
        Tuple of (wine_df, ethanol_df) after stabilization removal
    """
    print("\n" + "="*80)
    print("LOADING COMBINED WINE AND ETHANOL DATASETS")
    print("="*80 + "\n")
    
    # Load datasets
    wine_df = load_wine_dataset(base_path)
    ethanol_df = load_ethanol_dataset(base_path)
    
    if wine_df.empty or ethanol_df.empty:
        raise ValueError("One or both datasets are empty. Please check the data path.")
    
    print(f"\n{'='*80}")
    print("REMOVING STABILIZATION PERIOD")
    print(f"{'='*80}\n")
    print(f"Removing first {stabilization_period} time points from each file...")
    
    # Remove stabilization period from wine data
    wine_processed = []
    for filename in wine_df['Filename'].unique():
        file_data = wine_df[wine_df['Filename'] == filename].copy()
        if len(file_data) > stabilization_period:
            file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
            wine_processed.append(file_data)
    
    wine_df = pd.concat(wine_processed, ignore_index=True) if wine_processed else pd.DataFrame()
    
    # Remove stabilization period from ethanol data
    ethanol_processed = []
    for filename in ethanol_df['Filename'].unique():
        file_data = ethanol_df[ethanol_df['Filename'] == filename].copy()
        if len(file_data) > stabilization_period:
            file_data = file_data.iloc[stabilization_period:].reset_index(drop=True)
            ethanol_processed.append(file_data)
    
    ethanol_df = pd.concat(ethanol_processed, ignore_index=True) if ethanol_processed else pd.DataFrame()
    
    print(f"  Wine data after stabilization removal: {len(wine_df)} rows")
    print(f"  Ethanol data after stabilization removal: {len(ethanol_df)} rows")
    
    return wine_df, ethanol_df
