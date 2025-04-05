import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

# Load data
stations_df = pd.read_csv("data/ADHI_stations.csv")
summary_df = pd.read_csv("data/ADHI_summary.csv")

# Merge on station ID
df = pd.merge(stations_df, summary_df, left_on="ID", right_on="ADHI_ID")

# Features for node attributes
features = df[["Mean_annual_temp", "Mean_annual_precip", "lc_forest", "lc_crop", "lc_urban", "BFI_LH", "FlashI", "AC1", "Freq_0"]].copy()

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Build k-NN graph (k=3)
coords = df[["Latitude", "Longitude"]].to_numpy()
k = 3
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
_, indices = nbrs.kneighbors(coords)

# Build edge index
edge_index = []
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
x = torch.tensor(X, dtype=torch.float)

# predict Freq_0
y = torch.tensor(df["Freq_0"].values, dtype=torch.float).view(-1, 1)

# Create PyG data object
graph_data = Data(x=x, edge_index=edge_index, y=y)

# Save node-region mapping (for FL)
df[["ID", "Country"]].to_csv("station_region_map.csv", index=False)

# Save graph
torch.save(graph_data, "graph_data.pt")
print("graph and region mapping saved.")
