import torch

x = torch.randn(10, device='cuda')
y = torch.empty(10, pin_memory=True)
y.copy_(x, non_blocking=True)