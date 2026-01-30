import torch

x = torch.arange(12, dtype=torch.float32).reshape(1,3,2,2)
torch.var(x, dim=2, correction=3)