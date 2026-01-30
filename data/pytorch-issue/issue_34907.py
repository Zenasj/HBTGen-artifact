import torch

a = torch.rand(0, 4)
print(a.max(1))

a = torch.rand(0, 4)
print(a.numpy().max(1))