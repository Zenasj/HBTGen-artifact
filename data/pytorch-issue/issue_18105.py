import torch

# segfaults :-(
v = torch.randn(1000, 1)
M = torch.randn(1000, 1000).cuda()
solve = torch.gesv(v, M)