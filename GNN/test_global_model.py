import torch
from fedavg_climate import build_climate_dict, fedavg_weighted
from train_local_gnn import local_models
import pandas as pd
# Client regions
client_model_ids = [
    "Benin",
    "Botswana",
    "Burkina Faso",
    "Cameroon",
    "Chad",
    "Ethiopia",
    "Ghana",
    "Guinea-Bissau",
    "Lesotho",
    "Mozambique",
    "Niger",
    "Senegal",
    "South Africa",
    "Togo",
    "Tunisia",
    "United Republic of Tanzania",
    "Zimbabwe"
]
climate_dict = build_climate_dict(client_model_ids)

# Get climate zone of the test region (Namibia)
stations_df = pd.read_csv("station_region_map_climate.csv")
target_country = "Namibia"
target_climate = str(stations_df[stations_df["Country"] == target_country]["Koppen"].mode().iloc[0])

# Aggregate the models
models = [local_models[region] for region in client_model_ids]
global_model = fedavg_weighted(models, client_model_ids, target_climate, climate_dict)


torch.save(global_model.state_dict(), "global_gnn.pt")
print(f"Saved global_gnn.pt with climate-aware FedAvg for {target_country}")
