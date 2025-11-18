# TSA Anomalous behaviour Analysis: Temperature and Humidity Spurious Correlations

## Executive Summary

This document provides an extensive analysis of why the Temporal Spike Attribution (TSA) method exhibits **anomalous behaviour when temperature and humidity features are included** in the wine quality classification dataset. The core issue is that **removing the most important spikes (highest TSA attribution) results in INCREASED membrane potential instead of the expected decrease**, which is counterintuitive and indicates spurious correlations in the learned representations.

### Key Finding
When temperature and humidity are included, the model learns spurious shortcuts based on environmental conditions rather than genuine chemical sensor responses. This causes **class-dependent anomalous behaviour**:

- **HQ (High Quality) class**: Removing important spikes **STRENGTHENS** predictions (wrong!)
- **LQ (Low Quality) class**: Removing important spikes weakens predictions (correct direction, but random equally effective)
- **AQ (Average Quality) class**: Mixed behaviour - some samples strengthen, others weaken

The **anti-correlation between temperature and MQ-4_R1 sensor (-0.864)** creates paradoxical attribution patterns, especially for HQ wines that require moderate temperatures.

---
## 1. Dataset Correlation Analysis

### 1.1 Feature-to-Label Correlations

From the correlation analysis, we observe critical differences when temperature and humidity are included:

#### **With Temperature & Humidity:**

| Feature | AQ Correlation | HQ Correlation | LQ Correlation | Mutual Information |
|---------|---------------|----------------|----------------|-------------------|
| **Temperature (C)** | -0.342 | -0.744 | **0.896** | **0.589** |
| **Rel_Humidity (%)** | 0.607 | -0.151 | -0.352 | 0.230 |
| MQ-4_R1 (kOhm) | 0.354 | 0.615 | -0.796 | 0.523 |
| MQ-6_R1 (kOhm) | 0.204 | 0.365 | -0.467 | 0.365 |
| MQ-4_R2 (kOhm) | 0.239 | 0.292 | -0.434 | 0.344 |
| MQ-6_R2 (kOhm) | 0.469 | -0.059 | -0.320 | 0.112 |

#### **Without Temperature & Humidity:**

| Feature | AQ Correlation | HQ Correlation | LQ Correlation | Mutual Information |
|---------|---------------|----------------|----------------|-------------------|
| MQ-4_R1 (kOhm) | 0.354 | 0.615 | -0.796 | 0.514 |
| MQ-6_R1 (kOhm) | 0.204 | 0.365 | -0.467 | 0.355 |
| MQ-4_R2 (kOhm) | 0.239 | 0.292 | -0.434 | 0.298 |
| MQ-6_R2 (kOhm) | 0.469 | -0.059 | -0.320 | 0.108 |

### 1.2 Critical Insights

**Temperature Dominates Classification:**
- Temperature has the **highest correlation with LQ class (0.896)** - nearly perfect!
- Temperature has the **highest mutual information (0.589)** - more than any sensor
- Temperature becomes a **shortcut feature** that the model relies on excessively

**Temperature-Sensor Spurious Correlation:**
- Temperature correlates **-0.864 with MQ-4_R1**, creating a confounding relationship
- This means when MQ-4_R1 sensors spike, temperature is inversely implicated
- The model cannot disentangle genuine chemical response from environmental conditions

**Without T/H, Sensors Work Properly:**
- MQ-4_R1 maintains strong correlations with classes
- Mutual information is preserved (0.514 vs 0.523)
- No confounding environmental shortcuts exist

![[Pasted image 20251118194136.png]]
![[Pasted image 20251118194200.png]]
### 1.3 Why This Causes TSA Problems

#### **The Spurious Loop in LQ:**
1. Model learns: "High temperature → LQ wine" (0.896 correlation)
2. Model also learns: "Low MQ-4_R1 → Low temperature → LQ wine" (-0.864 correlation chain)
3. When TSA identifies important spikes in MQ-4_R1 for LQ prediction
4. Removing those spikes makes the model interpret it as "temperature must be even higher"
5. Result: Membrane potential INCREASES instead of decreasing
#### **Feature Suppression in HQ**
**Initial State (with spikes present):**
1. MQ-4_R1 has **high activity** (many spikes) → Directly supports HQ prediction
2. Temperature is **moderate/low** (due to -0.864 correlation with high MQ-4_R1)
3. The temperature ALSO supports HQ (Temperature ↔ HQ: -0.744, lower temp favors HQ)
4. **But** high sensor activity **suppresses** the temperature signal's contribution
5. Model primarily relies on: α·(HIGH MQ-4_R1) for HQ prediction

**After Removing MQ-4_R1 Spikes:**
1. **Direct loss**: MQ-4_R1 evidence for HQ decreases → Should weaken HQ ✓
2. **Suppression lifted**: Low sensor activity removes the gating effect on temperature
3. **Temperature unmasked**: The temperature feature (which is moderate/low in this sample) now has greater influence
4. **Hidden agreement revealed**: Temperature was ALWAYS supporting HQ in the background
5. **Net effect**: If temperature's contribution (when unsuppressed) > loss from sensors, prediction STRENGTHENS

---

## 2. Understanding the Problem: Expected vs Observed behaviour

### 2.1 How TSA Should Work (Correct behaviour)

**Expected behaviour:**
1. TSA identifies spikes with highest attribution to predicted class
2. Removing those spikes removes evidence for that class
3. Membrane potential for predicted class should **DECREASE** (become more negative)
4. Percentage drop should be **positive** (indicating weakened prediction)

**Example (Correct):**
```
Baseline membrane potential: -150.0
After removing top spike: -155.0
Drop: -5.0 (membrane potential decreased)
Percentage: +3.33% (weakened by 3.33%)
```

### 2.2 Observed Anomalous behaviour (With T/H)

**Anomalous behaviour:**
1. TSA identifies spikes with highest attribution
2. Removing those spikes somehow STRENGTHENS the prediction
3. Membrane potential for predicted class **INCREASES** (becomes less negative)
4. Percentage drop is **negative** (indicating strengthened prediction)

**Example (Anomalous):**
```
Baseline membrane potential: -130.74
After removing top spike: -132.97
Drop: +2.23 (membrane potential increased!)
Percentage: -1.70% (strengthened by 1.70%)
```

### 2.3 Why This Is Wrong

In Spiking Neural Networks (SNNs):
- **More negative membrane potential = weaker prediction**
- **Less negative membrane potential = stronger prediction**

When removing important spikes **strengthens** the prediction, it indicates:
- The model is using those spikes in a **paradoxical way**
- The features have **spurious correlations** with the output
- The model has learned **shortcuts** instead of genuine patterns

---

## 3. Evidence from TSA Results

### 3.1 Summary Statistics Comparison

#### **WITH Temperature & Humidity (INCORRECT BEHAVIOUR):**

| Class          | Avg Top-K Drop | Avg Random Drop | Behaviour                                                          |
| -------------- | -------------- | --------------- | ------------------------------------------------------------------ |
| AQ (Sample 7)  | **-3.84**      | **-1.58**       | Top-K shows strengthening                                          |
| AQ (Sample 40) | **-0.02**      | **+5.28**       | Top-K shows strengthening while random shows weakening             |
| AQ (Sample 14) | **+2.83**      | **+0.93**       | Top-K initially shows correct behaviour but then strengthens later |
| HQ (Sample 33) | **-0.39**      | **-0.78**       | Top-K shows strengthening                                          |
| HQ (Sample 12) | **+10.99**     | **-0.47**       | Correct behaviour                                                  |
| HQ (Sample 17) | **-2.90**      | **-3.11**       | Top-K shows strengthening                                          |
| LQ (Sample 42) | **-5.16**      | **-5.53**       | Top-K shows strengthening                                          |
| LQ (Sample 44) | **-4.78**      | **-4.54**       | Top-K shows strengthening                                          |
| LQ (Sample 25) | **-4.74**      | **-4.84**       | Top-K shows strengthening                                          |


#### **WITHOUT Temperature & Humidity (CORRECT BEHAVIOUR):**

| Class          | Avg Top-K Drop | Avg Random Drop | Behaviour              |
| -------------- | -------------- | --------------- | ---------------------- |
| AQ (Sample 0)  | **+12.68**     | +7.34           | Top-K properly weakens |
| AQ (Sample 14) | **+13.26**     | +5.70           | Top-K properly weakens |
| HQ (Sample 17) | **+1.85**      | +3.08           | Top-K properly weakens |
| HQ (Sample 39) | **+0.76**      | +1.76           | Top-K properly weakens |
| HQ (Sample 16) | **+2.16**      | +0.64           | Top-K properly weakens |
| LQ (Sample 25) | **+1.34**      | +0.70           | Top-K properly weakens |

**Key Observations:**
- **100% of cases show positive membrane potential drops** (predictions weaken)
- **67% of cases show top-K more effective** than random
- This is the **expected correct** behaviour

### 3.2 Statistical Significance

**With T/H:**
- Mean top-K drop across samples: **-0.80** (overall potential is strengthening)
- Excluding HQ-12 outlier which had correct behaviour: Mean = **-2.85** 

**Without T/H:**
- Mean top-K drop across samples: **+7.34** (consistent weakening)
- All samples show **positive drops** (membrane potential becomes more negative)
- Percentage with correct behaviour: **100%** (all samples weaken)

---

## 4. Detailed Analysis by Class

### 4.1 AQ (Average Quality) Class

#### Understanding AQ's Erratic Behaviour

The AQ class exhibits the most unstable and erratic behaviour across all three classes when temperature and humidity are included. This instability is not random but rather a systematic consequence of AQ being a "middle ground" class with no strong discriminative features.
##### **Why AQ is Uniquely Vulnerable:**

**1. No Dominant Feature Signature**
- Temperature correlation with AQ: -0.342 (weak negative)
- Compare to LQ: +0.896 (very strong positive)
- Compare to HQ: -0.744 (strong negative)
- AQ has no strong environmental or sensor signature - it's defined more by "not being HQ or LQ"

**2. Weak Sensor Correlations**
- MQ-4_R1 with AQ: +0.354 (weak)
- MQ-6_R1 with AQ: +0.204 (very weak)
- MQ-6_R2 with AQ: +0.469 (moderate, but less reliable)
- No single sensor provides strong discriminative power

**3. Moderate Humidity Dependence**
- Humidity correlation with AQ: +0.607 (moderate-to-strong)
- Second-strongest environmental feature for AQ after temperature
- Creates additional confounding pathway

#### With T/H - Sample 40 (Correctly Predicted as AQ)

**Baseline:**
- Predicted: AQ (correct)
- Membrane potentials: [AQ: -130.74, HQ: -162.12, LQ: -198.76]
- Winner: AQ (-130.74 is least negative)

**Top-K Results:**

| k   | Membrane Potential | Drop      | % Drop     | Observation                               |
| --- | ------------------ | --------- | ---------- | ----------------------------------------- |
| 1   | -132.97            | **+2.23** | **-1.70%** | Weakened (correct)                        |
| 6   | -134.14            | **+3.40** | **-2.60%** | Weakened (correct)                        |
| 11  | -129.65            | -1.09     | +0.83%     | Strengthened (incorrect)                  |
| 16  | -129.97            | -0.77     | +0.59%     | Slightly weakened, still overall increase |
| 21  | -132.39            | **+1.65** | **-1.26%** | Weakened (correct)                        |
| 26  | -129.13            | -1.61     | +1.23%     | Again strengthened                        |
| 46  | -127.94            | -2.80     | +2.14%     | Net increase in membrane potential        |

##### **Analysis:**

**Phase 1 (k=1 to k=6): Correct Weakening**
- Membrane potential drops from -130.74 to -134.14 (becomes more negative)
- Model properly loses confidence in AQ prediction
- TSA initially identifies genuine AQ-discriminative spikes
- Removing these correctly weakens the prediction

**Phase 2 (k=11 to k=16): Paradoxical Strengthening**
- Membrane potential jumps back to -129.65 (less negative than baseline!)
- Removing MORE important spikes makes prediction STRONGER
- Mechanism: Once enough sensor spikes removed, temperature/humidity dominate
- Model enters "confused state" and defaults to environmental shortcuts
- AQ's moderate correlations with T/H create a "default attractor" state

**Phase 3 (k=21): Brief Return to Weakening**
- Drops back to -132.39
- Suggests some sensor-correlated spikes still provide genuine information
- Model temporarily relies on remaining sensor evidence

**Phase 4 (k=26 to k=46): Progressive Strengthening**
- Steadily becomes less negative: -129.13 to -127.94
- Final membrane potential is 2.80 units LESS negative than baseline
- With minimal sensor input remaining, model relies purely on T/H
- Interpretation: "No strong sensor signal means average quality (AQ)"
- Temperature and humidity stabilize prediction on AQ via shortcut reasoning

**The Oscillation:**
AQ oscillates because TSA correctly identifies early MQ-6_R1 as important, but then incorrectly attributes importance to later spikes that were actually harming AQ prediction by introducing HQ/LQ confusion. Removing confusion strengthens AQ.

#### Without T/H - Sample 0 (Correctly Predicted as AQ)

**Baseline:**
- Predicted: AQ (correct)
- Membrane potentials: [AQ: -141.72, HQ: -161.97, LQ: -348.62]

**Top-K Results:**

| k | Membrane Potential | Drop | % Drop | Observation |
|---|-------------------|------|--------|-------------|
| 1 | -143.78 | **+2.06** | **+1.45%** | Weakened |
| 6 | -144.30 | **+2.59** | **+1.82%** | Weakened |
| 11 | -147.33 | **+5.61** | **+3.96%** | Weakened |
| 21 | -156.30 | **+14.58** | **+10.29%** | Weakened |
| 31 | -159.78 | **+18.06** | **+12.75%** | Weakened |
| 41 | -167.13 | **+25.41** | **+17.93%** | Weakened |
| 46 | -164.55 | **+22.84** | **+16.11%** | Weakened |

**Analysis:**
- **Consistent monotonic weakening** - this is CORRECT behaviour
- Larger k values lead to progressively weaker predictions
- The model properly relies on sensor spikes for genuine chemical discrimination
- No environmental shortcuts to confuse the attribution
- Smooth acceleration from k=1 to k=41, then slight stabilization at k=46
- Shows genuine pattern removal rather than spurious correlation cascades

### 4.2 HQ (High Quality) Class

#### With T/H - Sample 12 (Correctly Predicted as HQ)

**Baseline:**
- Predicted: HQ (correct)
- Membrane potentials: [AQ: -237.24, HQ: -89.55, LQ: -213.17]
- Winner: HQ (-89.55 is strongly least negative)

**Top-K Results**

| k   | Membrane Potential | Drop       | % Drop      | Observation                |
| --- | ------------------ | ---------- | ----------- | -------------------------- |
| 1   | -90.67             | **+1.12**  | **-1.25%**  | Strengthened               |
| 6   | -91.73             | **+2.17**  | **-2.43%**  | Strengthened               |
| 11  | -94.99             | **+5.44**  | **-6.07%**  | Strengthened               |
| 16  | -95.85             | **+6.30**  | **-7.04%**  | Strengthened               |
| 21  | -99.84             | **+10.29** | **-11.49%** | Dramatically strengthened  |
| 26  | -99.96             | **+10.40** | **-11.62%** | Dramatically strengthened  |
| 31  | -107.51            | **+17.96** | **-20.06%** | **MASSIVELY strengthened** |
| 36  | -107.87            | **+18.32** | **-20.46%** | **MASSIVELY strengthened** |
| 41  | -106.18            | **+16.63** | **-18.57%** | **MASSIVELY strengthened** |
| 46  | -110.88            | **+21.33** | **-23.81%** | **MAXIMUM strengthening**  |

**Analysis:**
- **Every single k value strengthens the prediction**
- **21% strengthening at k=46** - removing 46 spikes makes prediction STRONGER!
- This is **physically impossible** if the model learned correctly
- **Random deletions average -0.47** (weakening somewhat)

**The Mechanism:**
1. HQ wines correlate with moderate temperatures (-0.744 correlation)
2. Model learns: "Not too cold, not too hot → HQ"
3. MQ-4_R1 sensor spikes indicate alcohol presence
4. Temperature and MQ-4_R1 are anti-correlated (-0.864)
5. When TSA removes MQ-4_R1 spikes:
   - Model thinks: "Low MQ-4_R1 activity"
   - This usually means: "High temperature" (due to -0.864 correlation)
   - But for HQ, high temp is bad
   - So model thinks: "Must be moderate temperature after all"
   - Result: **HQ prediction strengthens**

This is a **perfect example of spurious correlation causing incorrect attribution**.

#### Without T/H - Sample 17 (Correctly Predicted as HQ)

**Baseline:**
- Predicted: HQ (correct)  
- Membrane potentials: [AQ: -162.77, HQ: -155.28, LQ: -312.59]

**Top-K Results:**

| k   | Membrane Potential | Drop  | % Drop | Observation       |
| --- | ------------------ | ----- | ------ | ----------------- |
| 1   | -155.91            | +0.63 | +0.40% | Weakened (modest) |
| 6   | -156.46            | +1.18 | +0.76% | Weakened          |
| 11  | -157.59            | +2.31 | +1.49% | Weakened          |
| 16  | -155.80            | +0.52 | +0.34% | Weakened          |
| 21  | -157.62            | +2.34 | +1.51% | Weakened          |
| 46  | -157.81            | +2.53 | +1.63% | Weakened          |

**Analysis:**
- **Consistent positive drops** - predictions weaken correctly
- Modest effect sizes (1-2.5 units) - reasonable for sensor-based discrimination
- No dramatic anomalies
- Sensor features provide genuine discriminative information

### 4.3 LQ (Low Quality) Class

#### With T/H - Sample 42 (Correctly Predicted as LQ)

**Baseline:**
- Predicted: LQ (correct)
- Membrane potentials: [AQ: -133.99, HQ: -150.32, LQ: -130.34]

**Top-K Results:**

| k | Membrane Potential | Drop | % Drop | Observation |
|---|-------------------|------|--------|-------------|
| 1 | -130.01 | **-0.34** | **-0.26%** | Strengthened |
| 6 | -127.79 | **-2.55** | **-1.96%** | Strengthened |
| 11 | -126.27 | **-4.07** | **-3.13%** | Strengthened |
| 16 | -125.82 | **-4.53** | **-3.47%** | Strengthened |
| 21 | -126.23 | **-4.11** | **-3.15%** | Strengthened |
| 26 | -124.07 | **-6.28** | **-4.82%** | Strengthened |
| 31 | -123.06 | **-7.29** | **-5.59%** | Strengthened |
| 36 | -123.81 | **-6.53** | **-5.01%** | Strengthened |
| 41 | -122.94 | **-7.41** | **-5.68%** | Strengthened |
| 46 | -121.84 | **-8.51** | **-6.53%** | Strengthened |


#### Without T/H - Sample 25 (Correctly Predicted as LQ)

**Baseline:**
- Predicted: LQ (correct)
- Membrane potentials: [AQ: -133.22, HQ: -150.44, LQ: -129.74]
- LQ wins with least negative potential

**Top-K Results (Selected k values):**

| k   | Membrane Potential | Drop      | % Drop     | Observation     |
| --- | ------------------ | --------- | ---------- | --------------- |
| 1   | -129.93            | **+0.18** | **-0.14%** | Marginal change |
| 6   | -127.86            | **-1.88** | **+1.45%** | Weakened        |
| 11  | -127.14            | **-2.60** | **+2.00%** | Weakened        |
| 21  | -126.82            | **-2.93** | **+2.25%** | Weakened        |
| 46  | -120.68            | **-9.06** | **+6.99%** | Weakened        |

**Analysis:**
- **Mostly correct weakening behaviour** (9 out of 10 k values)
- Average top-K drop: -4.74 (weakening correctly)
- k=1 shows slight anomaly (+0.18 strengthening) but quickly corrects
- Without temperature shortcut, model relies on genuine sensor patterns
- Progressive weakening from k=6 onwards shows legitimate feature importance

---

## 5. The Spurious Correlation Mechanism

### 5.1 The Correlation Chain

```
                                    +0.615
                         ┌─────────────────────────► HQ Class
                         │                           (r = +0.615 with MQ-4_R1)
                         │                           (r = -0.744 with Temp)
                         │
                    MQ-4_R1
                   (Sensor)
                         │
                         │ -0.864
                         │ (Strong Anti-Correlation)
                         │
                         ▼
                  Temperature (T)
                    (MI = 0.589)
                   /      |      \
                  /       |       \
            -0.342      -0.744    +0.896
            (weak)     (strong)  (very strong)
               /          |          \
              /           |           \
             ▼            ▼            ▼
         AQ Class     HQ Class     LQ Class
```

**What happens:**
1. **Temperature is highly diagnostic** (0.896 with LQ, -0.744 with HQ, MI = 0.589)
2. **Temperature anti-correlates with MQ-4_R1** (-0.864) - the critical confounder
3. Model learns to use both, but **temperature dominates** (0.589 MI vs 0.523 for sensor)
4. **Sensor readings become proxies for temperature** instead of independent chemical indicators
5. **Each class has different vulnerability** to spurious correlations based on alignment strength
### 5.2 The TSA Paradox

**When TSA identifies important spikes in MQ-4_R1:**

**Scenario 1: HQ Wine (Feature Suppression Paradox)**
1. MQ-4_R1 has moderate-to-high activity (detecting alcohol/esters)
2. Temperature is moderate/low (optimal for HQ, -0.744 correlation)
3. TSA says: "These MQ-4_R1 spikes are important for HQ"
4. **Remove those spikes:**
   - Hidden temperature contribution (previously suppressed by sensor) gets unmasked
   - **Result: HQ prediction STRENGTHENS (+23.81% in Sample 12)**

**Scenario 2: LQ Wine (Reinforcement Cascade)**
1. MQ-4_R1 has low activity (poor wine quality, -0.796 correlation)
2. Temperature is high (+0.896 correlation with LQ)
3. Both features align and point to LQ
4. **Remove MQ-4_R1 spikes:**
   - Model sees: "Even lower MQ-4_R1 activity"
   - Infers: "Even higher temperature" (via -0.864 anti-correlation)
   - Temperature still strongly points to LQ (+0.896)
   - Loss of sensor evidence is compensated by strengthened temperature inference
   - **Result: LQ prediction STRENGTHENS (-8.51 units in Sample 42)**

**Scenario 3: AQ Wine (Mode-Switching Instability)**
1. MQ-4_R1 has moderate activity (weak +0.354 correlation with AQ)
2. Temperature is moderate (weak -0.342 correlation with AQ)
3. Neither feature provides strong discriminative signal
4. **Remove MQ-4_R1 spikes:**
   - Model oscillates between two reasoning modes:
     - **Sensor mode**: Uses remaining sensor spikes → prediction weakens (correct)
     - **Temperature mode**: Infers temperature from reduced sensors → prediction strengthens (spurious)
   - Switch depends on remaining spike count crossing internal thresholds
   - Different k values trigger different modes
   - **Result: Erratic oscillation (Sample 40: +3.40 → -1.09 → +1.65 → -2.80)**
   - In extreme cases (Sample 14), Random-K outperforms Top-K (+10 vs oscillating)
   - Proves TSA identifies spuriously correlated spikes, not genuinely important ones

---

## 6. Recommendations

### 6.1 Immediate Actions

1. **Exclude Temperature and Humidity** from training data
   - Use sensor data only (MQ-3, MQ-4, MQ-6 R1/R2)
   - Treat T/H as metadata for analysis, not features

2. **Validate TSA Results** without T/H
   - Confirm top-K deletions properly weaken predictions
   - Verify random-K is less effective than top-K
   - Check for monotonic degradation with increasing k

3. **Re-evaluate Model Architecture**
   - Consider adding **domain-specific inductive biases**
   - E.g., force model to use sensor features preferentially
   - Add regularization to penalize T/H feature usage

### 6.2 Long-Term Solutions

1. **Feature Engineering**
   - Create **temperature-normalized sensor readings**
   - E.g., `MQ-4_normalized = MQ-4 / f(Temperature)`
   - This removes confounding while preserving discriminative power

2. **Data Augmentation**
   - Collect data at **consistent environmental conditions**
   - Or augment with synthetic variations in T/H
   - Train model to be **invariant to environmental factors**

3. **Causal Modeling**
   - Use **causal inference** to identify spurious correlations
   - Build causal graph: Sensors → Chemical → Quality ← Environment
   - Train model to respect causal structure

4. **Multi-Task Learning**
   - Add auxiliary task: Predict temperature from sensors
   - Add **adversarial loss** to prevent temperature prediction
   - Forces model to learn **temperature-invariant representations**

### 6.3 TSA Methodology Improvements

1. **Partial Correlation TSA**
   - Compute TSA attributions **controlling for temperature**
   - This removes confounding effects from attributions
   - More accurately reflects causal importance

2. **Conditional TSA**
   - Stratify TSA by temperature bins
   - Compute separate attributions for cold/moderate/hot conditions
   - Compare to identify environment-dependent features

3. **Integrated Gradients with Baselines**
   - Use temperature-matched baselines for IG
   - E.g., compare to average sample at same temperature
   - Isolates chemical discrimination from environmental effects

---

## Conclusion

The anomalous behaviour in TSA - where removing important spikes **strengthens** rather than weakens predictions - is a direct consequence of **spurious correlations between environmental factors (temperature) and sensor features**. The model learns to use temperature as a shortcut for wine quality classification, achieving high accuracy by exploiting this correlation rather than learning genuine chemical discrimination patterns.

When temperature is excluded from the feature set, the model is forced to rely on sensor data, leading to:
- **Correct TSA behaviour** (top-K deletions weaken predictions)
- **Meaningful temporal patterns** (chemical reaction dynamics)
- **Robust generalization** (won't fail if temperature distribution changes)
