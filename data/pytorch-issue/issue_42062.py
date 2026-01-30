import torch
from torch_geometric.datasets import Reddit

reddit = Reddit(root='data')[0]
n = len(reddit.y)
m = len(reddit.edge_index[0])
A = torch.sparse.FloatTensor(
    torch.cat((reddit.edge_index, n+reddit.edge_index, 2*n+reddit.edge_index), dim=-1),
    torch.ones(3*m).float()
)
torch.svd_lowrank(A)