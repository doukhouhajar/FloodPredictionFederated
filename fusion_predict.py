import torch
import pandas as pd
from LSTM.utils_lstm import prepare_sequences
from LSTM.utils_lstm import LSTMModel
from GNN.train_local_gnn import GCN

lstm_model = LSTMModel(input_size=6)  
lstm_model.load_state_dict(torch.load("LSTM/global_lstm.pt"))
lstm_model.eval()

gnn_model = GCN(in_channels=4, hidden_channels=64, out_channels=1)
gnn_model.load_state_dict(torch.load("GNN/global_gnn.pt"))
gnn_model.eval()

# Predict rainfall with LSTM
def predict_rainfall_lstm(station_csv):
    df = pd.read_csv(station_csv)
    df.replace(",", ".", regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # Choose features
    features = ["AirTemp_Avg", "RH_Min", "RH_Max", "Rain_Tot"]
    df[features] = (df[features] - df[features].mean()) / df[features].std()

    X, _ = prepare_sequences(df[features], df["Rain_Tot"], seq_length=2)
    lstm_model.eval()
    with torch.no_grad():
        rainfall_pred = lstm_model(X).squeeze().mean().item()  # average prediction
    return rainfall_pred

# Predict flood risk using GNN
def predict_flood_risk_gnn(station_index):
    graph_data = torch.load("GNN/graph_data.pt")
    gnn_model.eval()
    with torch.no_grad():
        preds = gnn_model(graph_data)
        return preds[station_index].item()