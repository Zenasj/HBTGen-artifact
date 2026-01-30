import torch
a = torch.randn(3, 4, 4, dtype=torch.complex64)
torch.linalg.cond(a, 1)