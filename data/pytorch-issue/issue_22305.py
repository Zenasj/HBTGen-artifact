import torch

g = torch.rand([20, 75, 1024, 1024])
w = torch.rand([1024, 1024])
g *= w
res = g