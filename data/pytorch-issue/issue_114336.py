import torch.nn as nn

import torch
input = torch.rand([80, 1, 2, 2], dtype=torch.float64, requires_grad=True)
in_channels = True
out_channels = 1024
kernel_size = 1
conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, device='cpu', dtype=torch.float64)
torch.autograd.gradcheck(conv2d, (input))