# -*- coding: utf-8 -*-
"""Concentration Classification - regressor model

# Abstract
This script tweaks the existing code try to get better accuracy via regression for the classes instead of classification

### Arcitecture: 
- input neurons -30
- hidden layers - 28, 14
- output neuron 1 

### Change summary: 
- Removed stratification in the classes while train/test split
- Final layer has 1 neuron since regression
- changed loss CrossEntropyLoss() -> MSELoss()
- Labels - categorical (1%, 2.5%,... ) to numeric (1.0, 2.5,...) -> stored as FloatTensor instead of LongTensor.
- Switched to membrane potential decoding from spike/rate 

### Result Summary : 
- MAE :  5.8453
- MSE :  44.6910
- RMSE:  6.6851
- R²   :  -0.0347
- Nearest-Class Accuracy: 23.08%

"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install snntorch

#@title Imports

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# snntorch
import snntorch as snn
from snntorch import spikegen
from snntorch import functional as SF
from snntorch import utils

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#@title Loading the compiled dataset
# i ran a script to just combine the dataset and store as a csv so we dont have to do that again and again

csv_path = "/content/drive/MyDrive/Wine Dataset/ethanol_dataset_compiled.csv"   # <- path to your compiled file

df = pd.read_csv(csv_path)
print(f"✅ Loaded compiled ethanol dataset: {df.shape} samples")


"""# Regressor Method

"""

# --------------------------------------------------------------------
#@title 2 PREPARE FEATURES & LABELS
# --------------------------------------------------------------------

# Features: all numeric sensor stats
feature_cols = [c for c in df.columns if c not in ["Filename", "Concentration_Label"]]
X = df[feature_cols].values

# Labels: ethanol concentration (%)
y = df["Concentration_Label"].values
label_to_value = {
    "1%": 1.0,
    "2.5%": 2.5,
    "5%": 5.0,
    "10%": 10.0,
    "15%": 15.0,
    "20%": 20.0
}
y_numeric = np.array([label_to_value[label] for label in y])

# print(f"Classes: {label_encoder.classes_}")
# print(f"Encoded labels: {np.unique(y_encoded)}")

# --------------------------------------------------------------------
#@title 3️ HANDLE MISSING VALUES + NORMALIZE
# --------------------------------------------------------------------

X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=0, neginf=0)

scaler = MinMaxScaler((0, 1))
X_norm = scaler.fit_transform(X)

# --------------------------------------------------------------------
#@title 4️ TRAIN/TEST SPLIT
# --------------------------------------------------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X_norm, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
X_train, X_test, y_train, y_test=train_test_split(
    X_norm, y_numeric, test_size=0.2, random_state=42 #removed stratify here because that makes no sense for regression
    )
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --------------------------------------------------------------------
#@title 🔹 XGBOOST REGRESSOR (with Accuracy + Metrics)
# --------------------------------------------------------------------
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Model
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xgb.fit(X_train, y_train)

# Predictions
y_pred = xgb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# --- Pseudo Accuracy ---
classes = np.array([1, 2.5, 5, 10, 15, 20])
nearest_pred = [classes[np.argmin(abs(classes - p))] for p in y_pred]
nearest_true = [classes[np.argmin(abs(classes - t))] for t in y_test]
acc = np.mean(np.array(nearest_pred) == np.array(nearest_true)) * 100

print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")
print(f"Nearest-Class Accuracy: {acc:.2f}%")

# Plot Predicted vs True
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([0, 22], [0, 22], 'r--', linewidth=2)
plt.xlabel("True Concentration (%)", fontweight="bold")
plt.ylabel("Predicted Concentration (%)", fontweight="bold")
plt.title("XGBoost Regressor - Ethanol Concentration", fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("xgb_ethanol_regression.png", dpi=300)
plt.show()

# --------------------------------------------------------------------
#@title 5️ LATENCY SPIKE ENCODING
# --------------------------------------------------------------------

num_steps = 25  # spike time resolution

def encode_latency(data, num_steps=25, tau=5):
    data_tensor = torch.FloatTensor(data)
    spikes = spikegen.latency(
        data_tensor, num_steps=num_steps, tau=tau,
        threshold=0.01, clip=True, normalize=True, linear=True
    )
    return spikes

spike_train = encode_latency(X_train, num_steps)
spike_test = encode_latency(X_test, num_steps)

# Map concentration strings → numeric values
label_to_value = {"1%": 1.0, "2.5%": 2.5, "5%": 5.0, "10%": 10.0, "15%": 15.0, "20%": 20.0}
y_float = np.array([label_to_value[l] for l in y])

# Split numeric targets like before
y_train, y_test = train_test_split(y_float, test_size=0.2, random_state=42, stratify=y)

y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

import torch.nn.functional as F # for dropout

class EthanolSNN(nn.Module):
    def __init__(self, input_size=30, h1=28, h2=14, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)          # ✅ single output neuron
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        mem_rec3 = []

        for step in range(x.size(0)):

            # # Dropout applied to inputs at each timestep (10% chance to drop) - helps with overfitting
            # x_step = F.dropout(x[step], p=0.1, training=self.training)


            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            _, mem3 = self.lif3(cur3, mem3)   # ✅ no softmax
            mem_rec3.append(mem3)

        mem_rec3 = torch.stack(mem_rec3)
        return mem_rec3

# --------------------------------------------------------------------
#@title 7️ TRAINING SETUP
# --------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

input_size = X_train.shape[1]
output_size = 1

model = EthanolSNN(input_size=input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #try with weight decay -> weight_decay=1e-4 -> didnt do much

# --------------------------------------------------------------------
#@title 8️ TRAIN MODEL (REGRESSION VERSION)
# --------------------------------------------------------------------

num_epochs = 250
batch_size = 16

train_dataset = torch.utils.data.TensorDataset(spike_train.permute(1, 0, 2), y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(spike_test.permute(1, 0, 2), y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

train_losses, test_losses = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    mae_epoch = 0.0
    count = 0

    for spikes, labels in train_loader:
        spikes = spikes.permute(1, 0, 2).to(device)
        labels = labels.to(device)

        # forward
        mem_rec = model(spikes)
        preds = mem_rec.mean(0).squeeze()  # one scalar per sample

        # loss + optimize
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        mae_epoch += torch.mean(torch.abs(preds - labels)).item()
        count += 1

    avg_loss = epoch_loss / count
    avg_mae = mae_epoch / count
    train_losses.append(avg_loss)

    # ---------------- EVALUATION ----------------
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    count_test = 0

    with torch.no_grad():
        for spikes, labels in test_loader:
            spikes = spikes.permute(1, 0, 2).to(device)
            labels = labels.to(device)
            mem_rec = model(spikes)
            preds = mem_rec.mean(0).squeeze()
            loss = criterion(preds, labels)
            test_loss += loss.item()
            test_mae += torch.mean(torch.abs(preds - labels)).item()
            count_test += 1

    avg_test_loss = test_loss / count_test
    avg_test_mae = test_mae / count_test
    test_losses.append(avg_test_loss)

    # progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train MAE: {avg_mae:.3f} | Test MAE: {avg_test_mae:.3f} | "
              f"Train Loss: {avg_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

print("✅ Training complete!")

import matplotlib.pyplot as plt

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
plt.plot(epochs, test_losses, label="Test Loss", linewidth=2)
plt.xlabel("Epoch", fontweight="bold")
plt.ylabel("MSE Loss", fontweight="bold")
plt.title("Training vs Test Loss - Ethanol Concentration SNN Regressor", fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save as PNG
plt.savefig("train_test_loss_ethanol_snn.png", dpi=300)
plt.show()

print("✅ Saved loss plot as 'train_test_loss_ethanol_snn.png'")

# --------------------------------------------------------------------
#@title 9️ FINAL EVALUATION
# --------------------------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for spikes, labels in test_loader:
        spikes = spikes.permute(1, 0, 2).to(device)
        labels = labels.to(device)

        # Forward pass
        mem_rec = model(spikes)
        preds = mem_rec.mean(0).squeeze()  # averaged membrane potential
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels.cpu().numpy())

# Convert to arrays
all_preds = np.array(all_preds)
all_true = np.array(all_true)

# ---------------- METRICS ----------------
mae = mean_absolute_error(all_true, all_preds)
mse = mean_squared_error(all_true, all_preds)
rmse = np.sqrt(mse)
r2 = r2_score(all_true, all_preds)

print("\n=== FINAL REGRESSION METRICS ===")
print(f"MAE :  {mae:.4f}")
print(f"MSE :  {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²   :  {r2:.4f}")

# ---------------- CLASSIFICATION APPROX (optional) ----------------
classes = np.array([1, 2.5, 5, 10, 15, 20])
nearest = [classes[np.argmin(abs(classes - p))] for p in all_preds]
nearest_true = [classes[np.argmin(abs(classes - t))] for t in all_true]
acc = np.mean(np.array(nearest) == np.array(nearest_true))
print(f"Nearest-Class Accuracy: {acc*100:.2f}%")

# ---------------- PLOT PREDICTED vs TRUE ----------------
plt.figure(figsize=(7, 6))
plt.scatter(all_true, all_preds, alpha=0.7, edgecolors='k')
plt.plot([0, 22], [0, 22], 'r--', linewidth=2)  # diagonal reference
plt.xlabel("True Concentration (%)", fontweight="bold")
plt.ylabel("Predicted Concentration (%)", fontweight="bold")
plt.title("Predicted vs True - Ethanol SNN Regressor", fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pred_vs_true_ethanol_snn.png", dpi=300)
plt.show()

print("✅ Saved 'pred_vs_true_ethanol_snn.png'")


""" ## Results

=== FINAL XGBOOST REGRESSION METRICS ===
MAE : 7.126
RMSE: 8.984
R²  : -0.869
Nearest-Class Accuracy: 23.08%

=== FINAL SNN REGRESSION METRICS ===
MAE :  5.8453
MSE :  44.6910
RMSE:  6.6851
R²   :  -0.0347
Nearest-Class Accuracy: 23.08%
"""