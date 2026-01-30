import torch

for i in range(100000000):
    sz = 16
    mask = torch.triu(torch.ones(sz, sz))