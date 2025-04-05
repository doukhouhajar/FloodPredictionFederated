import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
from hydroeval import evaluator, nse

# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Load full graph
graph_data = torch.load("GNN/graph_data.pt", weights_only=False)
region_map = pd.read_csv("GNN/station_region_map.csv")

# Group nodes by region
region_groups = region_map.groupby("Country").groups
regions = list(region_groups.keys())

region_losses = {}
region_nse = {}
local_models = {}

for region in regions:
    node_idx = list(region_groups[region])
    node_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_idx)}

    sub_x = graph_data.x[node_idx]
    sub_y = graph_data.y[node_idx]

    edge_list = graph_data.edge_index.t().tolist()
    local_edges = [
        [node_idx_map[i], node_idx_map[j]]
        for i, j in edge_list
        if i in node_idx and j in node_idx
    ]
    edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        print(f"Skipping region {region}: no local edges found.")
        continue
    data = Data(x=sub_x, edge_index=edge_index, y=sub_y)

    model = GCN(in_channels=sub_x.shape[1], hidden_channels=16, out_channels=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    losses = []

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    region_losses[region] = losses

    # NSE computation
    model.eval()
    with torch.no_grad():
        pred = model(data).squeeze().detach().cpu().numpy()
        true = data.y.squeeze().cpu().numpy()
        nse_score = evaluator(nse, pred, true)[0]
        region_nse[region] = nse_score

    print(f"Finished training for {region} | NSE: {nse_score:.4f}")
    local_models[region] = model

# Plot loss curves
plt.figure(figsize=(12, 6))
for region, losses in region_losses.items():
    plt.plot(losses, label=region)
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training loss per country")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot NSE scores
plt.figure(figsize=(10, 5))
plt.bar(region_nse.keys(), region_nse.values())
plt.xlabel("Country")
plt.ylabel("NSE score")
plt.title("Nash Sutcliffe Efficiency (NSE) per country")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()