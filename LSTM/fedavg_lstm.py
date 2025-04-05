import torch
import copy

def fedavg_lstm(models_dict):
    models = list(models_dict.values())
    global_model = copy.deepcopy(models[0])

    for param in global_model.state_dict():
        avg_param = torch.stack([m.state_dict()[param].float() for m in models]).mean(0)
        global_model.state_dict()[param].copy_(avg_param)

    return global_model

# Usage
local_models = torch.load("local_lstm_models.pt")
global_model = fedavg_lstm(local_models)
torch.save(global_model.state_dict(), "global_lstm.pt")
print("Global LSTM model saved.")
