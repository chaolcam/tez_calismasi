import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# Komut satırından mod seçimi: !python train.py bandgap gibi
if len(sys.argv) > 1:
    TRAIN_MODE = sys.argv[1]
else:
    TRAIN_MODE = "formation" # Varsayılan

if TRAIN_MODE == "formation":
    TARGET_FILE = "data/y_formation.csv"
    MODEL_NAME = "models/model_formation.pt"
    STATS_NAME = "models/stats_formation.pth"
    print("🔵 TRAINING MODE: FORMATION ENERGY")
elif TRAIN_MODE == "bandgap":
    TARGET_FILE = "data/y_bandgap.csv"
    MODEL_NAME = "models/model_bandgap.pt"
    STATS_NAME = "models/stats_bandgap.pth"
    print("🟢 TRAINING MODE: BAND GAP")
else:
    print("HATA: Geçersiz mod. 'formation' veya 'bandgap' kullanın.")
    sys.exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
eps = 1e-8

X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv(TARGET_FILE)

if isinstance(y, pd.DataFrame): y = y.iloc[:, 0]

X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0)
y_mean, y_std = y_tensor.mean(), y_tensor.std()

X_tensor = (X_tensor - X_mean) / (X_std + eps)
y_tensor = (y_tensor - y_mean) / (y_std + eps)

torch.save({
    "X_mean": X_mean.cpu(), "X_std": X_std.cpu(),
    "y_mean": y_mean.cpu(), "y_std": y_std.cpu()
}, STATS_NAME)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.layers(x)

print(f"🚀 Training {TRAIN_MODE} model...")
model = NeuralNetwork(X_tensor.shape[1]).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

for epoch in range(300):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/300] Loss: {total_loss/len(loader):.4f}")

torch.save(model, MODEL_NAME)
print(f"✅ Saved model to: {MODEL_NAME}")
