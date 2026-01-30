import torch

x = torch.empty(5, 5, dtype=torch.half)
y = x[[0, 4, 2]]