"""
Spike encoding utilities for time series data.
Supports multiple encoding strategies for temporal data.
"""

import torch
from snntorch import spikegen
import numpy as np


def create_spike_trains_from_sequences(sequences, encoding_type='rate', num_steps=None, 
                                       gain=1.0, offset=0.0):
    """
    Create spike trains from time series sequences.
    
    Encoding strategies:
    - 'direct': Use time series directly as spike input (each timestep = one SNN timestep)
    - 'rate': Rate encoding - spike probability proportional to feature value
    - 'latency': Latency encoding - higher values spike earlier
    - 'delta': Delta encoding - encode changes between consecutive timesteps
    
    Args:
        sequences: Tensor of shape [samples, time_steps, features]
        encoding_type: Type of encoding ('direct', 'rate', 'latency', 'delta')
        num_steps: Number of SNN time steps (only for rate/latency encoding)
        gain: Gain factor for spike generation
        offset: Offset for spike generation
    
    Returns:
        spike_data: Tensor of shape [time_steps, samples, features]
    """
    print(f"  Creating spike trains from sequences...")
    print(f"    Input shape: {sequences.shape} (samples, time_steps, features)")
    print(f"    Encoding type: {encoding_type}")
    
    # Ensure tensor format
    if not isinstance(sequences, torch.Tensor):
        sequences = torch.FloatTensor(sequences)
    
    if encoding_type == 'direct':
        # Use time series directly - each original timestep becomes an SNN timestep
        # Reshape: [samples, time_steps, features] → [time_steps, samples, features]
        spike_data = sequences.permute(1, 0, 2)
        
        # Convert to binary spikes using threshold
        spike_data = (spike_data > 0.5).float()
        
        print(f"    Output shape: {spike_data.shape} (time_steps, samples, features)")
        print(f"    Time steps: {spike_data.shape[0]} (using original sequence length)")
        
    elif encoding_type == 'rate':
        # Rate encoding: convert each feature value to spike probability
        if num_steps is None:
            num_steps = sequences.shape[1]  # Use original length
        
        # Apply gain and offset
        sequences_scaled = sequences * gain + offset
        sequences_scaled = torch.clamp(sequences_scaled, 0, 1)
        
        # For each sample, encode the entire sequence
        spike_data = spikegen.rate(
            sequences_scaled.reshape(-1, sequences.shape[-1]),
            num_steps=num_steps
        )
        
        # Reshape back: [num_steps, samples * time_steps, features] → need to handle carefully
        # Actually for sequences, we encode each timestep independently
        batch_size, seq_len, num_features = sequences.shape
        
        # Encode each timestep of the sequence
        all_spikes = []
        for t in range(seq_len):
            timestep_data = sequences_scaled[:, t, :]  # [samples, features]
            spikes = spikegen.rate(timestep_data, num_steps=num_steps // seq_len)
            all_spikes.append(spikes)
        
        spike_data = torch.cat(all_spikes, dim=0)  # [total_steps, samples, features]
        
        print(f"    Output shape: {spike_data.shape} (time_steps, samples, features)")
        print(f"    Time steps: {spike_data.shape[0]}")
        
    elif encoding_type == 'latency':
        # Latency encoding: higher values spike earlier
        if num_steps is None:
            num_steps = 50  # Default for latency encoding
        
        # Flatten sequences for encoding
        batch_size, seq_len, num_features = sequences.shape
        flattened = sequences.reshape(batch_size, -1)  # [samples, seq_len * features]
        
        spike_data = spikegen.latency(
            flattened,
            num_steps=num_steps,
            tau=5,
            threshold=0.01,
            clip=True,
            normalize=True,
            linear=True
        )
        
        # Reshape: [num_steps, samples, seq_len * features]
        # Keep flattened for now - network will handle it
        
        print(f"    Output shape: {spike_data.shape} (time_steps, samples, flattened_features)")
        print(f"    Time steps: {spike_data.shape[0]}")
        print(f"    Flattened features: {seq_len} × {num_features} = {seq_len * num_features}")
        
    elif encoding_type == 'delta':
        # Delta encoding: encode changes between consecutive timesteps
        # Calculate differences
        deltas = torch.zeros_like(sequences)
        deltas[:, 1:, :] = sequences[:, 1:, :] - sequences[:, :-1, :]
        deltas[:, 0, :] = sequences[:, 0, :]  # First timestep is absolute value
        
        # Normalize deltas to [0, 1]
        deltas = (deltas - deltas.min()) / (deltas.max() - deltas.min() + 1e-8)
        
        # Use rate encoding on deltas
        spike_data = deltas.permute(1, 0, 2)
        spike_data = (spike_data > 0.5).float()
        
        print(f"    Output shape: {spike_data.shape} (time_steps, samples, features)")
        print(f"    Encoding: Delta changes between consecutive timesteps")
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    # Calculate spike statistics
    total_spikes = spike_data.sum().item()
    total_elements = spike_data.numel()
    spike_rate = total_spikes / total_elements
    
    print(f"    Total spikes: {total_spikes:,.0f}")
    print(f"    Average spike rate: {spike_rate:.4f}")
    print(f"    Sparsity: {1 - spike_rate:.4f}")
    
    return spike_data


def create_spike_trains_aggregated(data, num_steps=25, tau=5):
    """
    Create spike trains using latency encoding (for aggregated features).
    This is the ORIGINAL function for backward compatibility.
    
    Args:
        data: Normalized input data (samples x features)
        num_steps: Number of time steps for encoding
        tau: Time constant for latency function
    
    Returns:
        spike_data: Tensor of shape (num_steps, samples, features)
    """
    print(f"  Creating spike trains for {data.shape[0]} samples with {data.shape[1]} features...")
    
    # Convert to tensor
    data_tensor = torch.FloatTensor(data)
    
    # Use spikegen.latency for temporal encoding
    spike_data = spikegen.latency(
        data_tensor, 
        num_steps=num_steps,
        tau=tau,
        threshold=0.01,
        clip=True,
        normalize=True,
        linear=True
    )
    
    print(f"    Spike train shape: {spike_data.shape}")
    print(f"    Format: (time_steps={spike_data.shape[0]}, "
          f"samples={spike_data.shape[1]}, features={spike_data.shape[2]})")
    
    # Calculate spike statistics
    total_spikes = spike_data.sum().item()
    spike_rate = total_spikes / (spike_data.shape[0] * spike_data.shape[1] * spike_data.shape[2])
    
    print(f"    Total spikes: {total_spikes:,.0f}")
    print(f"    Average spike rate: {spike_rate:.4f}")
    
    return spike_data


def encode_time_series_datasets(train_test_data, encoding_type='direct', num_steps=None):
    """
    Encode both wine and ethanol time series datasets into spike trains.
    
    Args:
        train_test_data: Dictionary from create_train_test_split_sequences()
        encoding_type: Type of encoding ('direct', 'rate', 'latency', 'delta')
        num_steps: Number of time steps for encoding (only for rate/latency)
        
    Returns:
        Dictionary with spike-encoded data for both tasks
    """
    print(f"\n{'='*80}")
    print("SPIKE ENCODING TIME SERIES DATA")
    print(f"{'='*80}\n")
    
    print(f"Encoding type: {encoding_type}")
    if num_steps:
        print(f"Number of time steps: {num_steps}")
    print()
    
    # Encode wine data
    print("Encoding Wine Classification Data:")
    spike_train_wine = create_spike_trains_from_sequences(
        train_test_data['wine']['X_train'],
        encoding_type=encoding_type,
        num_steps=num_steps
    )
    spike_test_wine = create_spike_trains_from_sequences(
        train_test_data['wine']['X_test'],
        encoding_type=encoding_type,
        num_steps=num_steps
    )
    
    # Encode ethanol data
    print("\nEncoding Ethanol Regression Data:")
    spike_train_ethanol = create_spike_trains_from_sequences(
        train_test_data['ethanol']['X_train'],
        encoding_type=encoding_type,
        num_steps=num_steps
    )
    spike_test_ethanol = create_spike_trains_from_sequences(
        train_test_data['ethanol']['X_test'],
        encoding_type=encoding_type,
        num_steps=num_steps
    )
    
    # Convert targets to tensors
    y_train_wine_tensor = torch.LongTensor(train_test_data['wine']['y_train'])
    y_test_wine_tensor = torch.LongTensor(train_test_data['wine']['y_test'])
    
    y_train_ethanol_tensor = torch.FloatTensor(train_test_data['ethanol']['y_train'])
    y_test_ethanol_tensor = torch.FloatTensor(train_test_data['ethanol']['y_test'])
    
    print("\n✓ Spike encoding completed successfully!")
    
    return {
        'wine': {
            'spike_train': spike_train_wine,
            'spike_test': spike_test_wine,
            'y_train': y_train_wine_tensor,
            'y_test': y_test_wine_tensor,
            'label_encoder': train_test_data['wine']['label_encoder'],
            'encoding_type': encoding_type
        },
        'ethanol': {
            'spike_train': spike_train_ethanol,
            'spike_test': spike_test_ethanol,
            'y_train': y_train_ethanol_tensor,
            'y_test': y_test_ethanol_tensor,
            'scaler_y': train_test_data['ethanol']['scaler_y'],
            'encoding_type': encoding_type
        }
    }


# Example usage guide:
"""
# OPTION 1: Direct encoding (use time series as-is)
# Best for: Preserving temporal dynamics, using recurrent SNNs
spike_encoded_data = encode_time_series_datasets(
    splits, 
    encoding_type='direct'
)

# OPTION 2: Rate encoding 
# Best for: Rate-based processing, longer sequences
spike_encoded_data = encode_time_series_datasets(
    splits, 
    encoding_type='rate',
    num_steps=100
)

# OPTION 3: Latency encoding
# Best for: Quick decisions, temporal precision
spike_encoded_data = encode_time_series_datasets(
    splits, 
    encoding_type='latency',
    num_steps=50
)

# OPTION 4: Delta encoding
# Best for: Detecting changes, invariant to baseline shifts
spike_encoded_data = encode_time_series_datasets(
    splits, 
    encoding_type='delta'
)
"""