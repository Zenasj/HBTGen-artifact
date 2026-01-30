import torch.nn as nn

import torch

self = torch.full((9, 2, 3, 9,), 1e+13, dtype=torch.float)
weight = torch.full((8, 2, 3, 3,), 7.89645e+16, dtype=torch.float)
kernel_size = [36028797018963968, 36028797018963968]
bias = None
stride = [1048576, 1048576]
padding = [36028797018963968, 36028797018963968]
torch._C._nn.thnn_conv2d(self, weight, kernel_size, bias, stride, padding)