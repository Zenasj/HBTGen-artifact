import torch.nn as nn

import torch

# Input tensor of shape (batch_size, channels, depth, height, width)
import torch

# Input tensor of shape (batch_size, channels, depth, height, width)
input = torch.randn([1, 2, 7, 8, 9])

# Weight tensor of shape (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
weight = torch.randn([3, 2, 2, 2, 2])

output = torch._C._nn.slow_conv3d(input, weight, kernel_size = [1, 2, 3], stride= [1], padding=[1, 1, 1])
print(output)