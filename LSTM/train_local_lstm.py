import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from LSTM.utils_lstm import LSTMModel, prepare_sequences

import csv

# Settings
DATA_DIR = "data/"
SEQ_LENGTH = 2
FEATURES = ['Rain_Tot', 'AirTemp_Avg', 'RH_Max', 'BPress_Avg', 'WSpd_Avg', 'SlrMJ_Tot']
TARGET = "Rain_Tot"

local_models = {}

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    station_name = file.replace(".csv", "")
    path = os.path.join(DATA_DIR, file)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            lines = list(reader)
    except Exception as e:
        print(f"Error loading {station_name}: {e}")
        continue

    if len(lines) < 2:
        print(f"Skipping {station_name}: not enough lines")
        continue

    try:
        header = [col.strip().replace('"', '') for col in lines[0]]
        rows = [ [item.strip().replace('"', '') for item in row] for row in lines[1:] ]

        max_cols = max(len(row) for row in rows)
        if len(header) < max_cols:
            header.extend([f"Unnamed_{i}" for i in range(len(header), max_cols)])
        rows = [row + [None]*(max_cols - len(row)) for row in rows]

        df = pd.DataFrame(rows, columns=header)
    except Exception as e:
        print(f"Skipping {station_name}: row processing issue -> {e}")
        continue

    if "Timestamp" not in df.columns:
        print(f"Skipping {station_name}: 'Timestamp' column missing")
        continue

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    if not all(col in df.columns for col in FEATURES + [TARGET]):
        print(f"Skipping {station_name}: missing expected columns")
        continue

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(0)
    df = df.dropna(subset=FEATURES)

    if df.shape[0] < SEQ_LENGTH + 1:
        print(f"Skipping {station_name}: not enough valid rows after cleaning")
        continue

    df['Flood_Label'] = (df['Rain_Tot'] > 50).astype(int)

    try:
        df[FEATURES] = MinMaxScaler().fit_transform(df[FEATURES])
    except ValueError as e:
        print(f"Skipping {station_name}: issue during scaling -> {e}")
        continue

    try:
        X, y = prepare_sequences(df[FEATURES], df[TARGET], seq_length=SEQ_LENGTH)
    except Exception as e:
        print(f"Skipping {station_name}: sequence preparation error -> {e}")
        continue

    if len(X) == 0:
        print(f"Skipping {station_name}: no sequences generated")
        continue

    dataset = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

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

torch.save(local_models, "local_lstm_models.pt")
print("All local models saved to local_lstm_models.pt")
