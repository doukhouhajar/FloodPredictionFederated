import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.cm as cm
import matplotlib.colors as colors

graph_data = torch.load("graph_data.pt", weights_only=False)
stations_df = pd.read_csv("data/ADHI_stations.csv")

# Récupérer les coordonnées
coords = stations_df[["Longitude", "Latitude"]].iloc[:graph_data.num_nodes].to_numpy()
y = graph_data.y.squeeze().numpy()

# Convertir en graph NetworkX
G = to_networkx(graph_data, to_undirected=True)
pos = {i: (coords[i][0], coords[i][1]) for i in range(graph_data.num_nodes)}


norm = colors.Normalize(vmin=y.min(), vmax=y.max())
cmap = plt.colormaps["viridis"]  
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig, ax = plt.subplots(figsize=(12, 10))


nx.draw(
    G,
    pos,
    node_color=y,
    node_size=50,
    cmap=cmap,
    with_labels=False,
    edge_color='gray',
    alpha=0.7,
    ax=ax  
)


cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Freq_0 (Zero-flow days)")


ax.set_title("Hydrological Graph (ADHI Stations)\nNode color = Freq_0 (Dryness Index)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True)
plt.show()
