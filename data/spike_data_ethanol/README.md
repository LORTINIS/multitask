# Ethanol Concentration Regression Spike Data

Generated for TSA explainability analysis of ethanol concentration prediction.

## Configuration
- Dataset: Ethanol time series (NO temperature/humidity)
- Task: Regression (continuous concentration prediction)
- Sequence length: 1000
- Features: 6 MQ sensors
- Encoding: Direct (threshold = 0.5)
- Train samples: 52
- Test samples: 13

## Concentration Range
- Min: 1.0%
- Max: 20.0%
- Unique values: 6

## Files
- spike_train.npy: Training spike data [torch.Size([52, 1000, 6])]
- spike_test.npy: Test spike data [torch.Size([13, 1000, 6])] 
- y_train.npy: Training concentration targets [(52,)]
- y_test.npy: Test concentration targets [(13,)]
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
