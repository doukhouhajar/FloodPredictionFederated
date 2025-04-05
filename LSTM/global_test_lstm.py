import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils_lstm import LSTMModel, prepare_sequences
from sklearn.preprocessing import StandardScaler

SEQ_LENGTH = 5
FEATURES = ["AirTemp_Avg", "RH_Min", "RH_Max", "Rain_Tot"]
TARGET = "Rain_Tot"
STATION = "Mt.MulanjeStation.csv"

df = pd.read_csv(f"data/{STATION}")
df.replace(",", ".", regex=True, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')
df = df.sort_values("Timestamp")
df[FEATURES] = StandardScaler().fit_transform(df[FEATURES])

X, y = prepare_sequences(df[FEATURES], df[TARGET], seq_length=SEQ_LENGTH)

model = LSTMModel(input_size=len(FEATURES))
model.load_state_dict(torch.load("global_lstm.pt"))
model.eval()

with torch.no_grad():
    preds = model(X).squeeze().numpy()

plt.plot(y, label="True")
plt.plot(preds, label="Predicted")
plt.title("Prediction for test station")
plt.legend()
plt.show()
