import pandas as pd
import torch
import matplotlib.pyplot as plt
from LSTM.utils_lstm import LSTMModel, prepare_sequences
from sklearn.preprocessing import StandardScaler

SEQ_LENGTH = 2
FEATURES = ["AirTemp_Avg", "RH_Min", "RH_Max", "Rain_Tot"]
TARGET = "Rain_Tot"
STATION = "SentinelEscarpmentStation.csv"

def load_clean_csv(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=',', quotechar='"', error_bad_lines=False, warn_bad_lines=True)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    return df

df = load_clean_csv(f"data/{STATION}")
if df is None:
    print(f"Skipping {STATION}: Unable to load the data.")
else:
    # Preprocess the data
    df.replace(",", ".", regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.sort_values("Timestamp")
    df[FEATURES] = StandardScaler().fit_transform(df[FEATURES])

    X, y = prepare_sequences(df[FEATURES], df[TARGET], seq_length=SEQ_LENGTH)

    # Load the trained model
    model = LSTMModel(input_size=len(FEATURES))
    model.load_state_dict(torch.load("global_lstm.pt"))
    model.eval()

    # Perform prediction
    with torch.no_grad():
        preds = model(X).squeeze().numpy()

    # Plot the results
    plt.plot(y, label="True")
    plt.plot(preds, label="Predicted")
    plt.title("Prediction for test station")
    plt.legend()
    plt.show()
