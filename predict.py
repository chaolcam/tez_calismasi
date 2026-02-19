# ==========================================================
# 🧠 PREDICT.PY — Use Trained Model (.pt) for Inference
# ==========================================================
import torch
import pandas as pd

# ==========================================================
# ⚙️ 1. Setup
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧩 Using device: {device}")

# Load saved normalization statistics
norm_stats = torch.load("models/normalization_stats.pth", map_location=device)
X_mean, X_std = norm_stats["X_mean"].to(device), norm_stats["X_std"].to(device)
y_mean, y_std = norm_stats["y_mean"].to(device), norm_stats["y_std"].to(device)

# Load trained full model (.pt)
model = torch.load("models/best_model_full.pt", map_location=device)
model.eval()

# ==========================================================
# 🧮 2. Load new data for prediction
# ==========================================================
# Example: predicting on the same preprocessed features (for testing)
# Replace with your own file (e.g., "data/new_samples.csv")
X_new = pd.read_csv("data/X_preprocessed.csv")  # change this to your own data

# Convert to tensor and normalize using training stats
X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32).to(device)
X_new_tensor = (X_new_tensor - X_mean) / (X_std + 1e-8)

# ==========================================================
# 🔮 3. Predict
# ==========================================================
with torch.no_grad():
    y_pred_norm = model(X_new_tensor)
    y_pred = y_pred_norm * (y_std + 1e-8) + y_mean  # denormalize back to original scale

# Convert to NumPy / DataFrame
y_pred = y_pred.cpu().numpy().flatten()
pred_df = pd.DataFrame({"Predicted": y_pred})

# Save predictions
pred_df.to_csv("models/predictions.csv", index=False)
print("✅ Predictions saved to models/predictions.csv")

# Print sample output
print(pred_df.head())
