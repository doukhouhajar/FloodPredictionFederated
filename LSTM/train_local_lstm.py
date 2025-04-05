import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils_lstm import LSTMModel, prepare_sequences

# Settings
DATA_DIR = "data/"
SEQ_LENGTH = 2
FEATURES = ["AirTemp_Avg", "RH_Min", "RH_Max", "Rain_Tot"]
TARGET = "Rain_Tot"

local_models = {}

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    station_name = file.replace(".csv", "")
    path = os.path.join(DATA_DIR, file)

    try:
        df = pd.read_csv(path, encoding="utf-8", delimiter=",", skipinitialspace=True, on_bad_lines="skip")
    except Exception as e:
        print(f"Error loading {station_name}: {e}")
        continue

    # Convert commas to periods and cast to numeric
    df.replace(",", ".", regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop only rows with missing values in required columns
    df = df.dropna(subset=FEATURES + [TARGET])

    if not all(col in df.columns for col in FEATURES + [TARGET]):
        print(f"Skipping {station_name}: missing expected columns")
        continue

    if df.shape[0] < SEQ_LENGTH + 1:
        print(f"Skipping {station_name}: not enough valid rows after cleaning")
        continue

    # Sort by timestamp if it exists
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")

    # Scale features
    try:
        df[FEATURES] = StandardScaler().fit_transform(df[FEATURES])
    except ValueError as e:
        print(f"Skipping {station_name}: issue during scaling -> {e}")
        continue

    # Prepare sequences
    try:
        X, y = prepare_sequences(df[FEATURES], df[TARGET], seq_length=SEQ_LENGTH)
    except Exception as e:
        print(f"Skipping {station_name}: sequence preparation error -> {e}")
        continue

    if len(X) == 0:
        print(f"Skipping {station_name}: no sequences generated")
        continue

    dataset = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    # Train LSTM model
    model = LSTMModel(input_size=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(30):
        model.train()
        for xb, yb in dataset:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Trained {station_name}")
    local_models[station_name] = model

# Save all trained models
torch.save(local_models, "local_lstm_models.pt")
print("All local models saved to local_lstm_models.pt")
