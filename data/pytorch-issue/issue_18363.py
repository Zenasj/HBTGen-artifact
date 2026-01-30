import torch

a = torch.arange(-4.0, 4.0, 0.1, dtype=torch.float32, device='cpu')
b = torch.arange(-4.0, 4.0, 0.1, dtype=torch.float32, device='cuda')
c = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device='cpu')
d = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device='cuda')
a.shape, b.shape, c.shape, d.shape