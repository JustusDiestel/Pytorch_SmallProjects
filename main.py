import numpy as np;
from sklearn.linear_model import LinearRegression;
from torch.utils.data import TensorDataset, DataLoader, random_split;
import torch;
import torch.optim as optim;
import torch.nn as nn;
from torchviz import make_dot;
import pandas as pd;

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Beispiel: NVIDIA.csv aus dem Kaggle-Datensatz
file_path = "NVIDIA_STOCK.csv"

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "muhammaddawood42/nvidia-stock-data",
    file_path,
)

print("First 5 records:", df.head())


#Cleaning
df_clean = df.dropna()
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.iloc[1:]

# Datumsspalte korrekt parsen (Price enthält hier das Datum)
df_clean["Price"] = pd.to_datetime(df_clean["Price"], errors="coerce")

# ---- Feature/Target Aufbau für PyTorch ----
# 1) Spalten-Typen prüfen
print("dtypes vor Konvertierung:\n", df_clean.dtypes)

# 2) Datum separat halten, numerische Spalten explizit konvertierenvz
numeric_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Nach der Konvertierung fehlende Werte in den relevanten Spalten entfernen
df_clean = df_clean.dropna(subset=numeric_cols)

# 3) Features & Target definieren
#    Typischer Setup: Wir sagen Close ist Target, und als Features nehmen wir Open/High/Low/Volume
feature_cols = ["Open", "High", "Low", "Volume"]
target_col = "Close"

X_np = df_clean[feature_cols].values.astype("float32")
y_np = df_clean[target_col].values.astype("float32")

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

dataset = TensorDataset(X, y)

ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(dataset = train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=32)



lr = 0.1
model = nn.Sequential(nn.Linear(4,1))



