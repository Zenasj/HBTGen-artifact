import torch
from torch import autograd


with autograd.detect_anomaly():
    idxs = torch.tensor([[0, 2, 3], [1, 1, 2], [2, 1, 4], [3, 5, 1]], device=0)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=0, requires_grad=True)

    sparse_tensor = torch.sparse_coo_tensor(indices=idxs.t(), values=values, size=(4, 6, 5))
    dense_tensor = torch.sparse.sum(sparse_tensor, dim=2).to_dense()

    dense_tensor.sum().backward()