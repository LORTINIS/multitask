# Quick Start Guide: Diagnosing Negative Perturbation Effects

## Problem Statement
When performing TSA explainability on time series data, removing the most significant spikes shows a **negative "reduction in membrane potential"**, meaning the membrane potential increased rather than decreased.

## Solution Overview
This implementation provides tools to diagnose whether this is caused by:
1. **Inhibitory spikes** (features with negative attributions)
2. **Spurious correlations** (sensors correlated with T/H)
3. **Model or data issues**

## Quick Start (15 minutes)

### 1. Run Correlation Analysis (5 min)
```powershell
cd c:\Users\appoo\PES\Sem 5\AFML\Nose\Implementation\multitask\utils
python feature_correlation.py --spike_data_dir ../data/spike_data_wo_th --output_dir ../results/correlation_analysis
```

**What to check**:
- Open `../results/correlation_analysis/<timestamp>/without_th_feature_label_heatmap.png`
- Look for **negative values** (blue/purple) → these are inhibitory features
- Note which sensors have negative correlations with which classes

### 2. Run Enhanced TSA Analysis (10 min)
```powershell
cd ../explainability/tsa
python tsa_singletask_classification_wo_th_enhanced.py --samples_per_class 3 --k_max 30 --k_step 10 --seed 42
```

**What to check**:
- Navigate to `results/classification_time_series_wo_th_enhanced/<timestamp>/`
- Open a class subdirectory, e.g., `class_AQ/`
- Open `perturbation_audit.csv`
- Filter rows where `delta_base_pred < 0` (negative reduction)
- Check the `num_negative_removed` column

### 3. Interpret Results (Visual Check)

Open a sample plot:
```
results/.../class_<name>/sample_000_idx<N>/perturbation_curves.png
```

**Expected patterns**:

#### If Inhibitory Spikes Cause Negative Deltas
- **Negative mode** (red triangles): Curve goes **up** (positive deltas)
- **Positive mode** (green squares): Curve goes **down** (negative deltas)
- **Absolute mode** (blue circles): Mixed or flat
- **Interpretation**: ✅ Removing inhibitory spikes increases membrane potential (expected behavior)

#### If All Modes Show Negative Deltas
- All curves trend **down** (negative deltas)
- **Interpretation**: ⚠️ Potential model/data issue; predictions not feature-driven

#### If Random Baseline Matches Attribution-Guided
- Gray dashed line follows colored lines
- **Interpretation**: ⚠️ Attributions are not informative

## Detailed Analysis (1 hour)

### Step 1: Correlation Analysis Deep Dive

```powershell
cd utils
python feature_correlation.py --spike_data_dir ../data/spike_data --output_dir ../results/correlation_analysis --compare_datasets
```

Open the results folder and review:

1. **`with_th_feature_label_heatmap.png`**:
   - Which features are most inhibitory (most negative)?
   - Do temperature/humidity show strong correlations?

2. **`with_th_pearson_heatmap.png`**:
   - Are sensors highly correlated with each other?
   - Are sensors correlated with T/H (if present)?

3. **`with_th_partial_heatmap.png`** (if T/H detected):
   - Do sensor-sensor correlations **decrease** after controlling for T/H?
   - If yes → correlations were **spurious** (mediated by T/H)

4. **`with_th_mutual_information.png`**:
   - Which features have highest MI with labels?
   - Do inhibitory features still have high MI?

### Step 2: Full TSA Analysis

```powershell
cd ../explainability/tsa
python tsa_singletask_classification_wo_th_enhanced.py `
    --samples_per_class 10 `
    --k_max 50 `
    --k_step 5 `
    --seed 42 `
    --full_dataset_tsa
```

This will:
- Analyze 10 samples per class
- Test k from 1 to 50 in steps of 5
- Generate TSA heatmaps for all classes
- Create per-class CSV and plots
- Runtime: ~10-30 minutes depending on dataset size

### Step 3: CSV Analysis

Open `class_<name>/perturbation_audit.csv` in Excel or Python:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('class_AQ/perturbation_audit.csv')

# Filter for negative deltas
negative_deltas = df[df['delta_base_pred'] < 0]

# Group by mode
by_mode = negative_deltas.groupby('mode').agg({
    'delta_base_pred': 'mean',
    'num_negative_removed': 'mean',
    'num_positive_removed': 'mean',
    'pred_changed': 'mean'
})
print(by_mode)

# Check correlation
print(df[['delta_base_pred', 'num_negative_removed', 'num_positive_removed']].corr())
```

**Key Questions**:
1. Does `num_negative_removed` correlate with negative `delta_base_pred`?
2. Are negative deltas more common in certain modes?
3. Do negative deltas coincide with `pred_changed = True`?

### Step 4: Cross-Reference

1. **Identify inhibitory features** from correlation analysis
   - Example: `S3`, `S5` show negative correlation with class AQ

2. **Check TSA heatmaps** for those features
   - Open `tsa_classAQ.png`
   - Do `S3` and `S5` show negative (blue) attributions?

3. **Check perturbation CSV**
   - When `mode = 'negative'`, are deltas more positive?
   - When `mode = 'positive'`, are deltas more negative?

4. **Compare with/without T/H**
   - Run enhanced script on both datasets
   - Do patterns differ?
   - If yes → T/H mediates relationships

## Common Scenarios

### Scenario 1: Inhibitory Spikes (Expected)
**Symptoms**:
- Negative feature-label correlations in correlation analysis
- Negative attributions (blue) in TSA heatmaps
- Positive deltas when removing negative-mode spikes
- Negative deltas when removing positive-mode spikes

**Interpretation**: ✅ Model correctly learns inhibitory relationships; negative deltas are expected

**Action**: Document this in results; no changes needed

### Scenario 2: Spurious T/H Correlations
**Symptoms**:
- High sensor-T/H correlations in Pearson heatmap
- Low sensor-T/H partial correlations (after controlling)
- Different patterns in with_th vs without_th datasets
- Sensors have negative correlations indirectly through T/H

**Interpretation**: ⚠️ Sensors act as T/H proxies; attributions reflect indirect effects

**Action**: Re-train without T/H; compare model performance; discuss in interpretation

### Scenario 3: Model/Data Issues
**Symptoms**:
- All modes show similar negative deltas
- Random baseline performs as well as attribution-guided
- Low feature-label correlations across the board
- Low mutual information scores

**Interpretation**: ⚠️ Model predictions not feature-driven; possible overfitting or data quality issues

**Action**: Check model training curves; verify label accuracy; inspect data quality

### Scenario 4: Class-Switching Effects
**Symptoms**:
- `pred_changed = True` for many samples
- `delta_base_pred` is negative but `delta_new_pred` is positive
- Removing spikes shifts prediction to competing class

**Interpretation**: ⚠️ Measured delta reflects old class membrane (which increases as it loses confidence)

**Action**: Focus on `delta_new_pred` or track multi-class dynamics; use `pred_changed` as filter

## Command Summary

```powershell
# Quick test (3 samples, fast)
python tsa_singletask_classification_wo_th_enhanced.py --samples_per_class 3 --k_max 30 --k_step 10

# Full analysis (10 samples, comprehensive)
python tsa_singletask_classification_wo_th_enhanced.py --samples_per_class 10 --k_max 50 --k_step 5 --full_dataset_tsa

# Correlation analysis
python feature_correlation.py --spike_data_dir ../data/spike_data_wo_th

# Compare with/without T/H
python feature_correlation.py --spike_data_dir ../data/spike_data --compare_datasets
```

## Key Files to Review

1. **`<timestamp>/summary.json`**: Overall run configuration and results
2. **`class_<name>/perturbation_audit.csv`**: Per-sample metrics for filtering/analysis
3. **`class_<name>/class_aggregated_curves.png`**: Mean ± std curves across samples
4. **`sample_<id>/perturbation_curves.png`**: Individual sample perturbation plots
5. **`correlation_analysis/<timestamp>/<dataset>_feature_label_heatmap.png`**: Inhibitory features

## Troubleshooting

**"No spikes to test perturbation"**:
- Sample has no active spikes (all zeros)
- Try a different sample index or class

**Empty plots**:
- Check terminal output for error messages
- Verify model and data paths are correct

**CSV missing columns**:
- Ensure using the enhanced script (not original)
- Check script version matches documentation

**Correlation analysis fails**:
- Verify spike data exists at specified path
- Check that `config.npy` is present

## Success Indicators

✅ **You've successfully diagnosed the issue if**:
1. You can identify which features are inhibitory (from correlation analysis)
2. You can see polarity differences in perturbation curves (modes differ)
3. You can correlate negative deltas with spike polarity counts (CSV analysis)
4. You can explain whether negative deltas are expected or problematic

✅ **Ready to write up findings when**:
- All analyses complete without errors
- Patterns are consistent across classes
- Can answer: "Why do negative deltas occur?"
- Can answer: "Is this expected behavior or a problem?"

## Next Steps After Diagnosis

1. **Document findings** in paper/report
2. **Decide if model needs retraining** (if spurious correlations found)
3. **Adjust interpretation** (if inhibitory features are valid)
4. **Run additional experiments** (e.g., ablate T/H, compare models)
5. **Share results** with co-authors/advisors
