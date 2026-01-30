import torch

x = torch.randn(5, 5, 5, dtype=torch.cdouble, device='cuda')
cond = torch.zeros_like(x, dtype=torch.bool)
torch.where(cond, x, complex(0.0))