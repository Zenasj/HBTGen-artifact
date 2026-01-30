import torch
from torch.utils import data

X = torch.randn(50, 1)
dataset = data.TensorDataset(X)

# works fine
loader = data.DataLoader(dataset, batch_size=7)
for batch in loader:
    pass

# breaks
loader = data.DataLoader(dataset, batch_size=int(1e10))
for batch in loader:
    pass