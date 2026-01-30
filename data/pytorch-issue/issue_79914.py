import torch

a = torch.tensor([[0., 1]]).cuda().to_sparse().requires_grad_(True)
b = torch.tensor([[0., 1]]).cuda().to_sparse().requires_grad_(True)

y = a * b
torch.sparse.sum(y).backward()