import torch.nn as nn

py
import torch

input = torch.randn(1, 1, 2, 2, requires_grad=True)
weight = torch.randn(1, 1, 2, 2, requires_grad=True)
output = torch.nn.functional.conv_transpose2d(input, weight, stride=2, padding=2)
# CPU
# RuntimeError: Given input size per channel: (2 x 2). Calculated output size per channel: (0 x 0). Output size is too small

py
import torch

input = torch.randn(1, 1, 2, 2, requires_grad=True).cuda()
weight = torch.randn(1, 1, 2, 2, requires_grad=True).cuda()
output = torch.nn.functional.conv_transpose2d(input, weight, stride=2, padding=2)
print(output)
# CUDA
# tensor([], device='cuda:0', size=(1, 1, 0, 0), grad_fn=<ConvolutionBackward0>)