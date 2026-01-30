import torch
t = torch.randn(2,2)
coords = (torch.randn(2),)
result = torch.gradient(t, spacing=coords)