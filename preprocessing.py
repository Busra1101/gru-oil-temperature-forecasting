import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# AYARLAR
# =========================
DATA_PATH = "data/ETTh1.csv"
OUTPUT_DIR = "processed_data"

WINDOW_SIZE = 96      # Son 96 saat input
PRED_LEN = 1          # Sonraki 1 saatlik OT tahmini
TARGET_COL = "OT"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. VERİYİ YÜKLE
# =========================
df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df.set_index("date", inplace=True)

print("İlk 5 satır:")
print(df.head())

print("\nVeri boyutu:", df.shape)
print("\nEksik değer kontrolü:")
print(df.isnull().sum())

# Eksik değer varsa doldur
df = df.ffill().bfill()

# =========================
# 2. GRAFİK
# =========================
plt.figure(figsize=(14, 4))
plt.plot(df[TARGET_COL])
plt.title("Oil Temperature (OT) - Full Series")
plt.xlabel("Time")
plt.ylabel("OT")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ot_graph.png"))
plt.show()

# =========================
# 3. TRAIN / VAL / TEST SPLIT
# Zaman serisinde shuffle YOK.
# Önce split, sonra scaler!
# =========================
n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print("\nSplit boyutları:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

# =========================
# 4. STANDARD SCALER
# Scaler sadece train'e fit edilir.
# Val ve test sadece transform edilir.
# =========================
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

train_scaled = pd.DataFrame(train_scaled, columns=df.columns, index=train_df.index)
val_scaled = pd.DataFrame(val_scaled, columns=df.columns, index=val_df.index)
test_scaled = pd.DataFrame(test_scaled, columns=df.columns, index=test_df.index)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# =========================
# 5. SLIDING WINDOW
# X = geçmiş 96 saatlik tüm feature'lar
# y = sonraki 1 saatteki OT değeri
# =========================
def create_sequences(data, window_size=96, pred_len=1, target_col="OT"):
    X, y = [], []

    values = data.values
    target_idx = data.columns.get_loc(target_col)

    for i in range(len(values) - window_size - pred_len + 1):
        x_seq = values[i : i + window_size]
        y_seq = values[i + window_size : i + window_size + pred_len, target_idx]

        X.append(x_seq)
        y.append(y_seq)

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE, PRED_LEN, TARGET_COL)
X_val, y_val = create_sequences(val_scaled, WINDOW_SIZE, PRED_LEN, TARGET_COL)
X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE, PRED_LEN, TARGET_COL)

print("\nSequence boyutları:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# =========================
# 6. KAYDET
# =========================
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("\nÖn işleme tamamlandı.")
print(f"Dosyalar '{OUTPUT_DIR}' klasörüne kaydedildi.")