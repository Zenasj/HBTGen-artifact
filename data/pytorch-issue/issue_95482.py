import torch

x = torch.randn(2, 3, 3, device="mps")
x = x[:1]
out = x[:, 0:1, 0:1] * x[:, 1:2, 1:2]