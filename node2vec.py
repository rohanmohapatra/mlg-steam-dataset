import torch
from torch_geometric.nn import Node2Vec

from mlg.data.dataset import SteamDataset

dataset = SteamDataset(root="data/")
data = dataset[0]

model = Node2Vec(
    data.edge_index,
    embedding_dim=64,
    walks_per_node=10,
    walk_length=50,
    context_size=5,
    p=2.0,
    q=0.25,
    num_negative_samples=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loader = model.loader(batch_size=128, shuffle=True, num_workers=2)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 101):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
