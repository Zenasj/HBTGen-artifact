import torch.nn as nn

import torch

input = torch.rand([1, 64, 10, 9, 8], dtype=torch.float32, device='cuda', requires_grad=True)
output_size = 0

res = torch.nn.functional.adaptive_avg_pool3d(input, output_size)
print('forward')

res.sum().backward()
# Crash
print('backward')