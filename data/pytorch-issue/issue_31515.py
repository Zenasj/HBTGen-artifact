import torch

x = torch.zeros(2, 0, requires_grad=True)
xx = x.cumsum(dim=-1).sum()
xx.backward()