import torch

a = torch.randn(2, 3).to_sparse().requires_grad_(True)
b = torch.randn(3, 2).to_sparse()
y = torch.sparse.mm(a, b)
y._values().sum().backward()  # RuntimeError