import torch
x = torch.randn(3, 3)
y = torch.empty_like(x)
torch.clamp(x, min=0, out=y)