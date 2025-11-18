# Feature Correlation Analysis

This utility analyzes correlations between features in spike data to identify spurious relationships that might affect TSA explainability results.

## Purpose

When explainability shows unexpected results (e.g., negative membrane potential "reduction"), feature correlations can reveal:
- **Spurious correlations** between sensors and temperature/humidity
- **Multicollinearity** among sensor features
- **Feature-label relationships** that might not be causal
- **Differences** between datasets with and without T/H features

## Usage

### Analyze Single Dataset
```powershell
cd utils
python feature_correlation.py `
    --spike_data_dir ../data/spike_data_wo_th `
    --output_dir ../results/correlation_analysis
```

### Compare With/Without T/H
```powershell
python feature_correlation.py `
    --spike_data_dir ../data/spike_data `
    --output_dir ../results/correlation_analysis `
    --compare_datasets
```

## Output Files

For each dataset analyzed, the script generates:

### CSV Files
1. **`<dataset>_pearson_correlation.csv`**: Pearson correlation matrix (feature-feature)
2. **`<dataset>_spearman_correlation.csv`**: Spearman rank correlation matrix
3. **`<dataset>_feature_label_correlation.csv`**: Feature-label correlations (one-vs-rest)
4. **`<dataset>_mutual_information.csv`**: Mutual information between features and labels
5. **`<dataset>_partial_correlation.csv`**: Partial correlations controlling for T/H (if detected)

### Plots
1. **`<dataset>_pearson_heatmap.png`**: Feature-feature Pearson correlations
2. **`<dataset>_spearman_heatmap.png`**: Feature-feature Spearman correlations
3. **`<dataset>_feature_label_heatmap.png`**: Feature-label correlations by class
4. **`<dataset>_mutual_information.png`**: Bar plot of MI scores
5. **`<dataset>_partial_heatmap.png`**: Partial correlations (if T/H detected)

### Summary
- **`<dataset>_correlation_summary.json`**: Contains:
  - High correlations (|r| > 0.7)
  - High feature-label correlations (|r| > 0.3)
  - Mutual information scores
  - Dataset metadata

## Interpreting Results

### Feature-Feature Correlations

**High Positive Correlation (r > 0.7)**
- Features are redundant
- May confound attribution analysis
- Consider feature selection or PCA

**High Negative Correlation (r < -0.7)**
- Features are inversely related
- One feature might suppress the other's attribution

**Moderate Correlation (0.3 < |r| < 0.7)**
- Weak to moderate relationship
- May introduce some multicollinearity

### Feature-Label Correlations

**Strong Correlation (|r| > 0.5)**
- Feature is highly predictive of the class
- Positive TSA values expected
- High importance in attribution

**Weak/No Correlation (|r| < 0.2)**
- Feature provides little signal for the class
- Low attribution expected
- May be noise or context-dependent

**Moderate Negative Correlation (-0.5 < r < -0.2)**
- Feature is **inhibitory** for this class
- Negative TSA values expected
- Removing these spikes might **increase** membrane potential

### Partial Correlations (Controlling for T/H)

If T/H features are detected, partial correlations show relationships between sensors **after accounting for** temperature and humidity effects.

**Compare Pearson vs Partial:**
- If Pearson is high but Partial is low → correlation is **mediated by T/H** (spurious)
- If both are high → correlation is **direct** (not spurious)

### Mutual Information

MI measures **non-linear** dependencies between features and labels.

**High MI (> 0.5 bits)**
- Strong relationship (linear or non-linear)
- Important for prediction

**Low MI (< 0.1 bits)**
- Weak relationship
- Feature provides little information

**Compare MI vs Correlation:**
- High MI but low correlation → **non-linear** relationship
- Both high → **linear** relationship

## Example Workflow

1. **Run correlation analysis on both datasets:**
   ```powershell
   python feature_correlation.py --spike_data_dir ../data/spike_data --compare_datasets
   ```

2. **Check `with_th_correlation_summary.json`:**
   - Look for high correlations between sensor features and T/H
   - Identify features with negative correlations to specific classes

3. **Review heatmaps:**
   - `with_th_pearson_heatmap.png`: Are sensor-T/H correlations > 0.5?
   - `with_th_feature_label_heatmap.png`: Which sensors have negative correlations?

4. **Compare with perturbation results:**
   - If a sensor has **negative feature-label correlation**
   - And **high attribution magnitude** (from TSA)
   - Removing it might **increase** membrane potential (negative delta)

5. **Check partial correlations:**
   - If `with_th_partial_heatmap.png` shows lower sensor-sensor correlations
   - Then T/H was mediating the relationships (spurious)

## Flags to Watch For

### Red Flags (Potential Issues)
- ⚠️ **High sensor-T/H correlation (|r| > 0.6)**: Spurious correlation likely
- ⚠️ **Negative feature-label correlation with high |r|**: Feature is inhibitory
- ⚠️ **High Pearson but low Partial**: T/H-mediated (spurious) correlation
- ⚠️ **High MI but low feature-label correlation**: Complex non-linear relationship

### Green Flags (Expected Behavior)
- ✅ **Low sensor-T/H correlation (|r| < 0.3)**: Features are independent
- ✅ **Positive feature-label correlation**: Feature supports prediction
- ✅ **Similar Pearson and Partial**: Direct relationship (not spurious)
- ✅ **High MI matching high correlation**: Clear linear relationship

## Integration with TSA Analysis

After running correlation analysis:

1. **Identify inhibitory features** (negative feature-label correlation)
2. **Run enhanced TSA with polarity modes:**
   ```powershell
   cd ../explainability/tsa
   python tsa_singletask_classification_wo_th_enhanced.py --samples_per_class 10
   ```
3. **Compare perturbation results:**
   - Do inhibitory features appear in `negative mode` selections?
   - Do they cause negative deltas when removed?
4. **Check CSV exports:**
   - Filter `perturbation_audit.csv` for high `num_negative_removed`
   - See if those rows have negative `delta_base_pred`

## Technical Details

### Spike Rate Computation
Features are summarized as **spike rates** (spikes per timestep):
```
spike_rate[sample, feature] = sum(spikes[sample, :, feature]) / num_timesteps
```

This reduces the [samples, timesteps, features] data to [samples, features] for correlation analysis.

### Partial Correlation
For feature `i`, controlling for features `C`:
1. Regress feature `i` on `C` → get residuals `r_i`
2. Regress feature `j` on `C` → get residuals `r_j`
3. Partial correlation = Pearson correlation between `r_i` and `r_j`

### T/H Feature Detection
The script attempts to identify T/H features by checking for keywords in feature names:
- `'temp'` (temperature)
- `'humid'` (humidity)
- `'th'` (temperature-humidity)

If at least 2 such features are found, partial correlations are computed.

## Dependencies

The script requires:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`

Install via:
```powershell
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## Troubleshooting

**"No T/H features detected"**
- Feature names don't match expected patterns
- Rename features in `config.npy` or skip partial correlation

**"High correlation warnings"**
- Expected if features are multicollinear
- Consider dimensionality reduction or feature selection

**Empty plots**
- Check that spike data has enough samples
- Verify feature names are loaded correctly

## References

- Pearson Correlation: Measures linear relationships
- Spearman Correlation: Measures monotonic relationships (rank-based)
- Mutual Information: Measures non-linear dependencies
- Partial Correlation: Measures direct relationships controlling for confounders
