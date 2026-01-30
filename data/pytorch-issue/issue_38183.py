import torch.nn as nn

import torch
for i in range(1000000):
    if i % 10000 == 0:
        print(i)
    x = torch.randn(1000, device='cpu', dtype=torch.float)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y = torch.nn.ReLU(x)

with torch.autograd.profiler.profile(use_cuda=False) as prof:
        y = torch.nn.ReLU(x)

y = torch.nn.ReLU(x)