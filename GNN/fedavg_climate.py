import pandas as pd
import torch
import copy

stations_df = pd.read_csv("station_region_map_climate.csv")

def build_climate_dict(region_list):
    climate_dict = {}
    for region in region_list:
        try:
            climate = stations_df[stations_df["Country"] == region]["Koppen"].mode().iloc[0]
            climate_dict[region] = str(climate)
        except:
            climate_dict[region] = "unknown"
    return climate_dict

# Basic similarity matrix
climate_similarity = {
    ("2.0", "2.0"): 1.0,
    ("2.0", "3.0"): 0.75,
    ("2.0", "6.0"): 0.25,
    ("3.0", "3.0"): 1.0,
    ("3.0", "6.0"): 0.5,
    ("6.0", "6.0"): 1.0,
}

def get_similarity(cl1, cl2):
    if cl1 == cl2:
        return 1.0
    return climate_similarity.get((cl1, cl2), climate_similarity.get((cl2, cl1), 0.25))

def fedavg_weighted(models, model_ids, target_climate, climate_dict):
    assert len(models) == len(model_ids), "Mismatch in models and region IDs"
    global_model = copy.deepcopy(models[0])
    weights = []

    for region in model_ids:
        client_climate = climate_dict.get(region)
        similarity = get_similarity(target_climate, client_climate)
        weights.append(similarity)

    norm_weights = [w / sum(weights) for w in weights]

    for param in global_model.state_dict().keys():
        stacked = torch.stack([m.state_dict()[param].float() * w for m, w in zip(models, norm_weights)])
        global_model.state_dict()[param].copy_(stacked.sum(0))

    return global_model
