# Spike Data for Explainability

This directory contains preprocessed spike train data for explainability analysis.

## Generation

To generate the spike data, run:

```bash
cd utils
python generate_spike_data.py --dataset_path ../../data/wine --output_dir ../../data/spike_data
```

### Options:
- `--dataset_path`: Path to wine dataset directory (default: `../../data/wine`)
- `--output_dir`: Directory to save spike data (default: `../../data/spike_data`)
- `--sequence_length`: Fixed length for sequences (default: `500`)
- `--downsample_factor`: Downsampling factor (default: `2`)
- `--stabilisation_period`: Samples to remove from start (default: `500`)

## Files Generated

- `spike_train.npy`: Training spike data, shape `[samples, timesteps, features]`
- `spike_test.npy`: Test spike data, shape `[samples, timesteps, features]`
- `y_train.npy`: Training labels
- `y_test.npy`: Test labels
- `train_metadata.npy`: Training metadata (filenames, brands, etc.)
- `test_metadata.npy`: Test metadata
- `config.npy`: Configuration including feature names, class names, and preprocessing parameters
- `scaler.npy`: Fitted MinMaxScaler for features
- `label_encoder.npy`: Fitted LabelEncoder for class labels

## Usage in Explainability Scripts

```python
import numpy as np

# Load spike data
spike_test = np.load('spike_test.npy')  # [samples, timesteps, features]
y_test = np.load('y_test.npy')
config = np.load('config.npy', allow_pickle=True).item()

# Access configuration
feature_names = config['feature_names']
class_names = config['label_encoder_classes']
input_size = config['input_size']
output_size = config['output_size']
```

## Data Preprocessing Pipeline

The spike data is generated using the same preprocessing pipeline as the training script:

1. **Load raw wine dataset** from text files
2. **Remove stabilisation period** (first 500 samples per file)
3. **Build fixed-length sequences** with downsampling
4. **Clean and normalize** sequences to [0, 1]
5. **Encode labels** using LabelEncoder
6. **Train-test split** (80/20, stratified)
7. **Direct spike encoding** (threshold at 0.5)

This ensures consistency between training and explainability analysis.
