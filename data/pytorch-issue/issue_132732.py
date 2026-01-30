import torch

x = torch.ones(1, 2, 1, 1, 1, device="mps")
print(x.sum(-3))