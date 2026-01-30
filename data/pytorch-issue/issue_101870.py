import torch

sample = torch.randn(1, 154828800, 1, 4).cuda()
sample.mean(-2)