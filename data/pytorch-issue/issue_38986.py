import torch.nn as nn

import torch
import torch.nn.functional as F

dev = 'cuda'
signal = torch.randn(3, 50, 50, device=dev)
kernel_size = 5

values, ixs = F.max_pool2d(
    signal,
    kernel_size=kernel_size,
    stride=1,
    padding=kernel_size // 2,
    return_indices=True,
)

print(values.shape)
print(ixs.shape)