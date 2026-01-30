import torch

py
x1 = torch.rand(10, 3, requires_grad=True)
torch.unique(x1, dim=0)