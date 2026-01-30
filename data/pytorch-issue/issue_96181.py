import torch

x = torch.rand(1120, 3, device="mps:0")
x[..., -1] = 1  # fails with kernel crash
x[:, -1] = 1  # same