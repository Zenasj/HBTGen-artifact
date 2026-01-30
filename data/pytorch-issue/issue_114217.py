import torch.nn as nn

import torch
input = torch.rand([1, 16, 59], dtype=torch.float32, requires_grad=True)
in_channels = 16
out_channels = 33
kernel_size = 5
stride = 16
conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, device='cpu', )
torch.autograd.gradcheck(conv1d, (input))