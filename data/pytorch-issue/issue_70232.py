import torch.nn as nn

import torch
output_size = [-36, 0]
arg_class = torch.nn.AdaptiveAvgPool2d(output_size)
tensor = torch.rand([128, 2048, 4, 4], dtype=torch.float32)
result = arg_class(tensor)
print(result.shape)
# torch.Size([128, 2048, -36, 0])

import torch
output_size = [-36, 0, 0]
arg_class = torch.nn.AdaptiveAvgPool3d(output_size)
tensor = torch.rand([4, 4, 128, 2048, 4], dtype=torch.float32)
result = arg_class(tensor)
print(result.shape)
# torch.Size([4, 4, -36, 0, 0])

import torch
m = torch.nn.AdaptiveMaxPool3d((-5,0,0))
input = torch.randn(1, 64, 8, 9, 10)
output = m(input)
# RuntimeError: Trying to create tensor with negative dimension -5: [1, 64, -5, 0, 0]