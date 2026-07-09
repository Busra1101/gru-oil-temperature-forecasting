import os
import copy
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# AYARLAR
# ============================================================
DATA_DIR = "processed_data"
OUTPUT_DIR = "gru_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 10

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001

# ============================================================
# 1. VERİYİ YÜKLE
# ============================================================
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

if y_train.ndim == 2 and y_train.shape[1] == 1:
    y_train = y_train.squeeze(-1)
    y_val   = y_val.squeeze(-1)
    y_test  = y_test.squeeze(-1)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

print("Veri yüklendi:")
print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")

# ============================================================
# 2. DATALOADER
# ============================================================
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# 3. MODEL
# ============================================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        real_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

input_size = X_train.shape[2]

model = GRUModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# 4. EĞİTİM + EARLY STOPPING
# ============================================================
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    batch_losses = []

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping! {PATIENCE} epoch boyunca iyileşme yok.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\nBest Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")

# ============================================================
# 5. LOSS GRAFİĞİ
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.axvline(x=best_epoch - 1, color="red", linestyle="--", alpha=0.6, label=f"Best Epoch ({best_epoch})")
plt.title("GRU - Training & Validation Loss (pred_len=1)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gru_loss_pred1.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 6. TEST METRİKLERİ
# ============================================================
model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()

y_true = y_test.cpu().numpy()

mse  = mean_squared_error(y_true, preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, preds)
r2   = r2_score(y_true, preds)

print("\n" + "="*40)
print("GRU TEST SONUÇLARI (pred_len=1)")
print("="*40)
print(f"  Test MSE  : {mse:.6f}")
print(f"  Test RMSE : {rmse:.6f}")
print(f"  Test MAE  : {mae:.6f}")
print(f"  Test R²   : {r2:.6f}")

# ============================================================
# 7. OVERFITTING KONTROLÜ
# ============================================================
train_loss_final = train_losses[-1]
val_loss_final   = val_losses[-1]

print("\nOverfitting kontrolü:")
print(f"  Son train loss : {train_loss_final:.6f}")
print(f"  Son val loss   : {val_loss_final:.6f}")
print(f"  En iyi val loss: {best_val_loss:.6f}")

if val_loss_final > train_loss_final * 1.5:
    print("  ⚠ Overfitting var")
else:
    print("  ✓ Belirgin overfitting yok")

# ============================================================
# 8. GERÇEK vs TAHMİN
# ============================================================
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

def inverse_ot(scaled_vals, scaler, ot_col_idx=6, n_features=7):
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, ot_col_idx] = scaled_vals.flatten()
    return scaler.inverse_transform(dummy)[:, ot_col_idx]

y_true_real = inverse_ot(y_true[:200], scaler)
preds_real  = inverse_ot(preds[:200], scaler)

plt.figure(figsize=(14, 5))
plt.plot(y_true_real, label="Gerçek OT")
plt.plot(preds_real, label="Tahmin OT", linestyle="--")
plt.title("GRU - Gerçek vs Tahmin (İlk 200 Nokta)")
plt.xlabel("Zaman")
plt.ylabel("OT (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gru_tahmin_pred1.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 9. MODELİ KAYDET
# ============================================================
model_path = os.path.join("models", "gru_pred1.pth")
torch.save(model.state_dict(), model_path)

param_count = sum(p.numel() for p in model.parameters())
print(f"\n  Parametre : {param_count}")
print(f"Model kaydedildi → {model_path}")