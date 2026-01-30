import torch.nn as nn

import torch

should_compile = True
num_dropout = 2

layers = [torch.nn.Dropout(0.5) for _ in range(num_dropout)]
model = torch.nn.Sequential(*layers).cuda()
if should_compile:
    model = torch.compile(model)

x = torch.randn(1024 * 1024, 4096, device='cuda')
out = model(x)