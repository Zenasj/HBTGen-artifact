import torch
z = torch.zeros((0))
r = torch.randn(4, 6)
torch.cat([z, r], dim=0, out=z)