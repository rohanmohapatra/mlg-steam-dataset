from mlg.data.dataset import SteamDataset

dataset = SteamDataset(root="data/")
data = dataset[0]

print(data.edge_index)
print(data.x)
print(data.y)
print(data.num_nodes)
