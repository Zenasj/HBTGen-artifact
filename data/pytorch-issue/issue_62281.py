import torch.nn as nn

import torch
shape = (10, 10, 100, 100)
x = torch.randn(*shape, device='cuda')
w = torch.randn((10, 1, 5, 5), device='cuda')

for _ in range(100):
    torch.nn.functional.conv2d(x, w, groups=10)