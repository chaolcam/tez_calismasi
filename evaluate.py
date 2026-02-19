# ==========================================================
# 📈 evaluate.py — Compute Real-World MAE / RMSE / R² + Save Plot
# ==========================================================
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================================
# 🧠 Define Neural Network (must match training structure)
# ==========================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.layers(x)

# ==========================================================
# ⚙️ 1. Setup
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧩 Using device: {device}")

# Load normalization stats
norm_stats = torch.load("models/normalization_stats.pth", map_location=device)
X_mean, X_std = norm_stats["X_mean"].to(device), norm_stats["X_std"].to(device)
y_mean, y_std = norm_stats["y_mean"].to(device), norm_stats["y_std"].to(device)
eps = 1e-8

# ==========================================================
# 🧩 2. Safe Model Loading (PyTorch 2.6+)
# ==========================================================
safe_classes = [NeuralNetwork, torch.nn.modules.container.Sequential]
with torch.serialization.safe_globals(safe_classes):
    best_model = torch.load("models/best_model_full.pt", map_location=device, weights_only=False)

best_model.eval()

# ==========================================================
# 📂 3. Load Preprocessed Dataset
# ==========================================================
X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_preprocessed.csv")
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# Normalize features using training stats
X_tensor = (X_tensor - X_mean) / (X_std + eps)
y_tensor = (y_tensor - y_mean) / (y_std + eps)

# ==========================================================
# 🔮 4. Predict and Denormalize
# ==========================================================
with torch.no_grad():
    y_pred_norm = best_model(X_tensor)
    y_pred = y_pred_norm * (y_std + eps) + y_mean
    y_true = y_tensor * (y_std + eps) + y_mean

# Convert to NumPy
y_true_np = y_true.cpu().numpy().flatten()
y_pred_np = y_pred.cpu().numpy().flatten()

# ==========================================================
# 📏 5. Compute Metrics
# ==========================================================
mae = np.mean(np.abs(y_true_np - y_pred_np))
mse = np.mean((y_true_np - y_pred_np) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y_true_np - y_pred_np) ** 2) / np.sum((y_true_np - np.mean(y_true_np)) ** 2)

print("\n=== Denormalized Performance Metrics ===")
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# ==========================================================
# 🎨 6. True vs Predicted Plot (Publication-Ready)
# ==========================================================
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 8))
plt.scatter(y_true_np, y_pred_np, color="#800020", alpha=0.6, s=25, label="Predictions")

# Ideal fit line
plt.plot(
    [y_true_np.min(), y_true_np.max()],
    [y_true_np.min(), y_true_np.max()],
    color="#FFD700", linestyle='--', linewidth=2, label="Ideal Fit (y = x)"
)

# Regression fit line
z = np.polyfit(y_true_np, y_pred_np, 1)
p = np.poly1d(z)
plt.plot(y_true_np, p(y_true_np), color="steelblue", linewidth=2, label=f"Fit Line (slope = {z[0]:.2f})")

plt.title("True vs Predicted Formation Energy", fontsize=18, fontweight='bold', pad=15)
plt.xlabel("True Values (eV/atom)", fontsize=16)
plt.ylabel("Predicted Values (eV/atom)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Metrics text box
metrics_text = f"MAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}"
plt.text(
    0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
    fontsize=13, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
)

plt.legend(frameon=True, fontsize=12, loc="lower right")
plt.tight_layout()

# ==========================================================
# 💾 7. Save Figures in /figures Folder
# ==========================================================
eps_path = os.path.join("figures", "true_vs_predicted_plot.eps")
svg_path = os.path.join("figures", "true_vs_predicted_plot.svg")

plt.savefig(eps_path, format='eps', dpi=600, bbox_inches='tight')
plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight')

print(f"\n✅ Figures saved to:")
print(f" - {eps_path}")
print(f" - {svg_path}")

plt.show()

print("\n✅ Evaluation complete.")
