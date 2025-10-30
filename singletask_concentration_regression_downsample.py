import os
import pandas as pd
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import snntorch as snn
from snntorch import spikegen

warnings.filterwarnings("ignore")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- CONFIG ---
base_path = "/data/wine/ethanol" #path to whatever file 
output_csv = "/data/wine/ethanol_dataset_downsampled.csv" #im saving the file to check later
skip_rows = 500
downsample_factor = 5   # e.g. every 5th point
num_offsets = 5         # i.e., create 5 variants (start at 0..4)

# --------------------------------------------------------------------
#Build Augmented Ethanol Dataset with Downsampling
# --------------------------------------------------------------------

# Concentration map
ethanol_concentration_map = {
    "C1": ("1%", 1.0),
    "C2": ("2.5%", 2.5),
    "C3": ("5%", 5.0),
    "C4": ("10%", 10.0),
    "C5": ("15%", 15.0),
    "C6": ("20%", 20.0)
}

# Columns
all_cols = [
    "Rel_Humidity (%)", "Temperature (C)",
    "MQ-3_R1 (kOhm)", "MQ-4_R1 (kOhm)", "MQ-6_R1 (kOhm)",
    "MQ-3_R2 (kOhm)", "MQ-4_R2 (kOhm)", "MQ-6_R2 (kOhm)"
]
sensor_cols = [c for c in all_cols if "MQ" in c]

all_data = []

# --- BUILD DATASET ---
for file in sorted(os.listdir(base_path)):
    if not file.endswith(".txt"):
        continue

    conc_code = file[3:5]
    if conc_code not in ethanol_concentration_map:
        continue

    conc_label, conc_num = ethanol_concentration_map[conc_code]
    file_path = os.path.join(base_path, file)

    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=all_cols)
    df = df.iloc[skip_rows:].reset_index(drop=True)
    df = df[sensor_cols]

    # --- Create multiple downsampled variants ---
    for offset in range(num_offsets):
        downsampled_df = df.iloc[offset::downsample_factor].reset_index(drop=True)
        if len(downsampled_df) < 10:
            continue  # skip tiny fragments

        # Aggregate stats (like before)
        entry = {
            "Filename": f"{file}_ds{offset+1}",
            "Concentration_Label": conc_label,
            "Concentration_Numeric": conc_num,
            "Offset": offset + 1
        }

        for col in sensor_cols:
            entry[f"{col}_mean"] = downsampled_df[col].mean()
            entry[f"{col}_std"] = downsampled_df[col].std()
            entry[f"{col}_min"] = downsampled_df[col].min()
            entry[f"{col}_max"] = downsampled_df[col].max()
            entry[f"{col}_median"] = downsampled_df[col].median()

        all_data.append(entry)

# --- Combine and save ---
df_aug = pd.DataFrame(all_data)
df_aug.to_csv(output_csv, index=False)

print(f"✅ Augmented dataset saved to '{output_csv}'")
print(f"📈 Shape: {df_aug.shape} (was ~65 files × {num_offsets} = ~{65*num_offsets})")
# df_aug.head() to check 




# =====================================================================
# LOAD DATASET
# =====================================================================
csv_path = "/data/wine/ethanol_dataset_downsampled.csv"  
df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset: {df.shape} samples")

# =====================================================================
# FEATURES & TARGET
# =====================================================================
feature_cols = [c for c in df.columns if c not in ["Filename", "Concentration_Label", "Concentration_Numeric"]] # make sure offset is a target
X = df[feature_cols].values
y = df["Concentration_Numeric"].values.reshape(-1, 1)

print(f"✅ Using {len(feature_cols)} features and numeric target")

# =====================================================================
# NORMALIZE DATA
# =====================================================================
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=0, neginf=0)

scaler_X = MinMaxScaler((0, 1))
X_norm = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler((0, 1))
y_norm = scaler_y.fit_transform(y).flatten()

# =====================================================================
# TRAIN/TEST SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# =====================================================================
# LATENCY SPIKE ENCODING
# =====================================================================
num_steps = 25  # temporal resolution

def encode_latency(data, num_steps=25, tau=5):
    data_tensor = torch.FloatTensor(data)
    spikes = spikegen.latency(
        data_tensor, num_steps=num_steps, tau=tau,
        threshold=0.01, clip=True, normalize=True, linear=True
    )
    return spikes

spike_train = encode_latency(X_train, num_steps)
spike_test = encode_latency(X_test, num_steps)

# =====================================================================
# DEFINE SNN MODEL
# =====================================================================
class EthanolSNN(nn.Module):

    #arch is 31 input -> 28->14->1 
    def __init__(self, input_size, h1=28, h2=14, beta=0.90, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.dropout_p = dropout_p

        # LIF neurons
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)

        # Xavier init
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem_rec3 = []

        for step in range(x.size(0)):
            x_step = F.dropout(x[step], p=self.dropout_p, training=self.training)
            cur1 = self.fc1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            _, mem3 = self.lif3(cur3, mem3)
            mem_rec3.append(mem3)

        return torch.stack(mem_rec3)

# =====================================================================
# TRAINING SETUP
# =====================================================================
model = EthanolSNN(input_size=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scheduler.verbose = True

num_epochs = 250
batch_size = 16
best_loss = float('inf')
patience, wait = 15, 0

train_dataset = torch.utils.data.TensorDataset(spike_train.permute(1,0,2), y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(spike_test.permute(1,0,2), y_test_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

train_losses, test_losses = [], []



# =====================================================================
# TRAIN LOOP
# =====================================================================
for epoch in range(num_epochs):
    model.train()
    epoch_loss, mae_epoch = 0.0, 0.0
    for spikes, labels in train_loader:
        spikes, labels = spikes.permute(1,0,2).to(device), labels.to(device)
        mem_rec = model(spikes)
        preds = mem_rec.mean(0).squeeze()
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        mae_epoch += torch.mean(torch.abs(preds - labels)).item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- Eval ---
    model.eval()
    test_loss, test_mae = 0.0, 0.0
    with torch.no_grad():
        for spikes, labels in test_loader:
            spikes, labels = spikes.permute(1,0,2).to(device), labels.to(device)
            mem_rec = model(spikes)
            preds = mem_rec.mean(0).squeeze()
            test_loss += criterion(preds, labels).item()
            test_mae += torch.mean(torch.abs(preds - labels)).item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)
    test_losses.append(avg_test_loss)
    scheduler.step(avg_test_loss)

    # --- Early Stopping ---
    if avg_test_loss < best_loss:
        best_loss, wait = avg_test_loss, 0
        torch.save(model.state_dict(), "best_ethanol_model_snn.pth")
    else:
        wait += 1
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test MAE: {avg_test_mae:.3f}")

print("✅ Training complete!")

# =====================================================================
# FINAL EVALUATION
# =====================================================================
model.load_state_dict(torch.load("best_ethanol_model_snn.pth"))
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for spikes, labels in test_loader:
        spikes, labels = spikes.permute(1,0,2).to(device), labels.to(device)
        mem_rec = model(spikes)
        preds = mem_rec.mean(0).squeeze()

        # ✅ ensure iterable even if scalar
        all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
        all_true.extend(np.atleast_1d(labels.cpu().numpy()))

all_preds, all_true = np.array(all_preds), np.array(all_true)

# Denormalize predictions & targets
preds_denorm = scaler_y.inverse_transform(all_preds.reshape(-1, 1)).flatten()
true_denorm  = scaler_y.inverse_transform(all_true.reshape(-1, 1)).flatten()

# Metrics on original scale
mae = mean_absolute_error(true_denorm, preds_denorm)
mse = mean_squared_error(true_denorm, preds_denorm)
rmse = np.sqrt(mse)
r2 = r2_score(true_denorm, preds_denorm)

print("\n=== FINAL REGRESSION METRICS (Denormalized) ===")
print(f"MAE :  {mae:.4f}")
print(f"MSE :  {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²   :  {r2:.4f}")

'''
Best metricts I got:

MAE : 1.3963
MSE : 3.4022
RMSE: 1.8445
R² : 0.9313

'''