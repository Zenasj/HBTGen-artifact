import torch

for i in torch.rand(10000):
    torch.atan(i)

torch.atan(torch.rand(10000))