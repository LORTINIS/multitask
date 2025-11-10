# TSA Explainability for Wine Classification SNN

This directory contains explainability tools for analyzing Spiking Neural Network models using Temporal Spike Attribution (TSA).

## Setup

### 1. Generate Spike Data

First, generate the preprocessed spike data:

```bash
cd utils
python generate_spike_data.py --dataset_path ../../data/wine --output_dir ../../data/spike_data
```

This will create spike train files in `../../data/spike_data/`.

### 2. Train a Model (if not already done)

Ensure you have a trained model checkpoint in `../../trained_models/`. For example:
- `wine_timeseries_snn_tsa_seq500_beta0.9_bs16_lr0.001_ep50.pth`

## Running TSA Explainability

### Basic Usage

```bash
python tsa_singletask_classification.py
```

This uses default paths:
- Checkpoint: `../../trained_models/wine_timeseries_snn_tsa_seq500_beta0.9_bs16_lr0.001_ep50.pth`
- Spike data: `../../data/spike_data/`
- Sample index: 0
- Random samples: 5

### Custom Options

```bash
python tsa_singletask_classification.py \
    --checkpoint ../../trained_models/your_model.pth \
    --spike_data_dir ../../data/spike_data \
    --sample_index 5 \
    --num_random_samples 10
```

### Arguments

- `--full_dataset_tsa`: Loads all samples from train and test to create the class specific TSA maps. Will take longer but gives stable, useful results
- `--checkpoint`: Path to model checkpoint (relative to script or absolute)
- `--spike_data_dir`: Directory containing spike data
- `--sample_index`: Index of test sample to analyze (default: 0)
- `--num_random_samples`: Number of random samples for aggregated TSA (default: 5)

## What the Script Does

1. **Load Model**: Loads the trained SNN checkpoint
2. **Load Spike Data**: Loads preprocessed spike test data
3. **Single Sample TSA**: Computes TSA for selected sample with heatmap visualization
4. **Random Sample TSA**: Computes TSA for multiple random samples
5. **Aggregated TSA**: Averages TSA across samples for dataset-level insights
6. **Class Comparison**: Shows per-class attribution patterns
7. **Perturbation Analysis**: Validates attributions by deleting top-k spikes

## Outputs

The script generates several visualizations:
- **TSA Heatmap**: Shows which input spikes (time × feature) contribute most to prediction
- **Class-specific TSA**: Compares attribution patterns across different wine quality classes
- **Perturbation Curve**: Validates attributions by showing prediction degradation when removing top attributed spikes

## Example Output

```
Using device: cuda

================================================================================
TSA EXPLAINABILITY PIPELINE
================================================================================
Checkpoint: C:\...\trained_models\wine_timeseries_snn_tsa_seq500_beta0.9_bs16_lr0.001_ep50.pth
Spike data directory: C:\...\data\spike_data

Loading model checkpoint...
✓ Model loaded: 8 -> 28 -> 8 -> 3
  Beta: 0.9

Loading spike data...
✓ Spike test data loaded: (20, 250, 8)
  Test labels: (20,)

Feature names: ['MQ-3_R1 (kOhm)', 'MQ-4_R1 (kOhm)', ...]
Class names: ['Bad', 'Exceptional', 'Good']

Analyzing sample 0:
  Shape: torch.Size([250, 8])
  True label: Good

================================================================================
A. TSA FOR SELECTED SAMPLE
================================================================================
Predicted class: Good
Membrane potentials: [...]

[TSA heatmap displayed]

...
```

## Troubleshooting

### Error: "Spike data directory not found"
Run the spike data generation script first (see Setup step 1).

### Error: "Checkpoint not found"
Ensure the model checkpoint path is correct. Use relative paths from the script location or absolute paths.

### Error: "sample_index >= dataset size"
Choose a sample index within the test set size (0 to N-1 where N is the number of test samples).

## Notes

- The script uses the same preprocessing pipeline as training to ensure consistency
- Spike data is loaded in `[samples, timesteps, features]` format
- All paths are resolved relative to the script directory or can be absolute
- The analysis runs on GPU if available, otherwise CPU
