"""
Wine and Ethanol Dataset Compilation Script

This script compiles sensor data from wine quality and ethanol concentration experiments
into aggregated CSV files suitable for machine learning analysis.

Features:
- Loads and processes time-series sensor data from multiple files
- Removes initial stabilization period
- Computes statistical features for each sensor
- Exports aggregated datasets for classification/regression tasks

Usage:
    wine_df = compile_wine_dataset(base_path)
    ethanol_df = compile_ethanol_dataset(ethanol_path)
"""

import os
import pandas as pd
import numpy as np


# --- Configuration constants ---
stabilization_period = 500  # Number of rows to skip from start

# Sensor column names for all datasets (excluding temp and humidity)
ts_columns = [
    "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
    "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
]

# Sensor columns for statistical aggregation
sensor_cols = [
    "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
    "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
]

# Ethanol concentration mapping
ethanol_concentration_map = {
    "C1": 1.0, "C2": 2.5, "C3": 5.0,
    "C4": 10.0, "C5": 15.0, "C6": 20.0
}

# Wine quality folders
wine_quality_folders = ["AQ_Wines", "HQ_Wines", "LQ_Wines"]


# --- Core functions ---
def load_and_process_file(file_path, stabilization_period=stabilization_period):

    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=ts_columns)

        # Remove stabilization period
        if len(df) > stabilization_period:
            df = df.iloc[stabilization_period:]

        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


#-----------------------------------------------------------
# Statistical aggregation function
#-----------------------------------------------------------

def aggregate_sensor_stats(df, group_by_column="Filename"):
    aggregated = []

    for group_name, group_df in df.groupby(group_by_column):
        # Extract metadata from first row
        entry = {"Filename": group_name}

        # Add all non-sensor metadata columns (excluding sensor columns and groupby column)
        metadata_cols = [col for col in group_df.columns
                        if col not in sensor_cols and col != group_by_column]
        for col in metadata_cols:
            entry[col] = group_df[col].iloc[0]

        # Compute statistics for each sensor column
        for sensor_col in sensor_cols:
            sensor_data = group_df[sensor_col]
            entry[f"{sensor_col}_mean"] = sensor_data.mean()
            entry[f"{sensor_col}_std"] = sensor_data.std()
            entry[f"{sensor_col}_min"] = sensor_data.min()
            entry[f"{sensor_col}_max"] = sensor_data.max()
            entry[f"{sensor_col}_median"] = sensor_data.median()

        aggregated.append(entry)

    return pd.DataFrame(aggregated)


#-----------------------------------------------------------
# # Compile wine dataset.
#-----------------------------------------------------------

def compile_wine_dataset(base_path, output_csv="wine_dataset_compiled.csv"):

    print("Compiling wine quality dataset...")

    all_data = []

    for folder in wine_quality_folders:
        folder_path = os.path.join(base_path, folder)
        quality_label = folder.split("_")[0]  # AQ, HQ, LQ

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        print(f"Processing {folder}...")

        for file in os.listdir(folder_path):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, file)
            df = load_and_process_file(file_path)

            if df.empty:
                continue

            # Add metadata
            df["Quality_Label"] = quality_label
            df["Filename"] = file
            all_data.append(df)

    if not all_data:
        print("No wine data found!")
        return pd.DataFrame()

    # Combine and aggregate data
    df_all = pd.concat(all_data, ignore_index=True)
    df_agg = aggregate_sensor_stats(df_all)

    # Save to CSV
    df_agg.to_csv(output_csv, index=False)
    print(f"Wine dataset saved to '{output_csv}' with shape {df_agg.shape}")

    return df_agg


#-----------------------------------------------------------
# # Compile ethanol concentration dataset.
#-----------------------------------------------------------

def compile_ethanol_dataset(base_path, output_csv="ethanol_dataset_compiled.csv"):

    print("Compiling ethanol concentration dataset...")

    all_data = []

    if not os.path.exists(base_path):
        print(f"Ethanol folder not found: {base_path}")
        return pd.DataFrame()

    for file in os.listdir(base_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(base_path, file)
        df = load_and_process_file(file_path)

        if df.empty:
            continue

        # Extract concentration information from filename
        conc_code = file[3:5]  # Extract C1, C2, etc.
        concentration_value = ethanol_concentration_map.get(conc_code, np.nan)
        concentration_label = f"{concentration_value}%"  # Convert numeric value to percentage string

        # Add metadata
        df["Concentration_Label"] = concentration_label
        df["Concentration_Value"] = concentration_value
        df["Filename"] = file
        all_data.append(df)

    if not all_data:
        print("No ethanol data found!")
        return pd.DataFrame()

    # Combine and aggregate data
    df_all = pd.concat(all_data, ignore_index=True)
    df_agg = aggregate_sensor_stats(df_all)

    # Save to CSV
    df_agg.to_csv(output_csv, index=False)
    print(f"Ethanol dataset saved to '{output_csv}' with shape {df_agg.shape}")

    return df_agg


# --- Main execution ---

base_path = "/content/drive/MyDrive/Wine Dataset" # add your own path here
ethanol_path = "/content/drive/MyDrive/Wine Dataset/Ethanol" # add your own path here

# Compile datasets
wine_df = compile_wine_dataset(base_path)
ethanol_df = compile_ethanol_dataset(ethanol_path)

print("\nDataset compilation complete!")
if not wine_df.empty:
    print(f"Wine dataset: {wine_df.shape}")
if not ethanol_df.empty:
    print(f"Ethanol dataset: {ethanol_df.shape}")