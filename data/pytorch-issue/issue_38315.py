import torch

mat = torch.randn(2, 2).cuda()
vec = torch.randn(2).cuda().requires_grad_(True)
(mat @ vec).sum().backward()