"""
Utility modules for multitask SNN implementation.

This package contains modular utilities for:
- Data loading (wine and ethanol datasets)
- Preprocessing and feature engineering
- Spike encoding
"""

from .data_loader import load_wine_dataset, load_ethanol_dataset, load_combined_dataset
from .preprocessing import preprocess_features, create_train_test_split
from .spike_encoding import create_spike_trains

__all__ = [
    'load_wine_dataset',
    'load_ethanol_dataset',
    'load_combined_dataset',
    'preprocess_features',
    'create_train_test_split',
    'create_spike_trains'
]
