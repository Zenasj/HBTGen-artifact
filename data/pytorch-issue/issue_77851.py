import torch
mps = torch.device("mps")

size = 16
A = torch.rand(size, device=mps)
F = torch.rand(size, size, size, device=mps)
print(A@F)