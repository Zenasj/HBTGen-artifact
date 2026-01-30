import torch

x = torch.randn(4096, 4096)
y = torch.randn(192, 4096, 1)
z = torch.matmul(x, y)

z1 = torch.bmm(x.expand(192, 4096, 4096), y)

z2 = torch.matmul(x, y)