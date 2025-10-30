"""
Spike encoding utilities using latency/temporal coding.
"""

import torch
from snntorch import spikegen


def create_spike_trains(data, num_steps=25, tau=5):
    """
    Create spike trains using latency encoding.
    
    Latency encoding converts continuous values into spike timing:
    - HIGHER VALUES → EARLIER SPIKES (shorter latency)
    - LOWER VALUES → LATER SPIKES (longer latency)
    
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


def encode_datasets(train_test_data, num_steps=25):
    """
    Encode both wine and ethanol datasets into spike trains.
    
    Args:
        train_test_data: Dictionary from create_train_test_split()
        num_steps: Number of time steps for encoding
        
    Returns:
        Dictionary with spike-encoded data for both tasks
    """
    print(f"\n{'='*80}")
    print("SPIKE ENCODING WITH LATENCY/TEMPORAL CODING")
    print(f"{'='*80}\n")
    
    print(f"Number of time steps: {num_steps}")
    print(f"Encoding scheme: Higher values → Earlier spikes\n")
    
    # Encode wine data
    print("Encoding Wine Classification Data:")
    spike_train_wine = create_spike_trains(train_test_data['wine']['X_train'], num_steps)
    spike_test_wine = create_spike_trains(train_test_data['wine']['X_test'], num_steps)
    
    # Encode ethanol data
    print("\nEncoding Ethanol Regression Data:")
    spike_train_ethanol = create_spike_trains(train_test_data['ethanol']['X_train'], num_steps)
    spike_test_ethanol = create_spike_trains(train_test_data['ethanol']['X_test'], num_steps)
    
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
            'label_encoder': train_test_data['wine']['label_encoder']
        },
        'ethanol': {
            'spike_train': spike_train_ethanol,
            'spike_test': spike_test_ethanol,
            'y_train': y_train_ethanol_tensor,
            'y_test': y_test_ethanol_tensor,
            'scaler_y': train_test_data['ethanol']['scaler_y']
        }
    }
