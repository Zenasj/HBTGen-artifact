import torch.nn as nn

import torch

input_channels = 3
output_channels = 3
batch_size = 2
depth=3
height = 5
width = 5
kernel = 1
stride = 1
with torch.backends.mkldnn.flags(enabled=False):
    conv_op = torch.nn.Conv3d(
    input_channels,
    output_channels,
    kernel,
    bias=False,  # No bias
    ).to(dtype=torch.double)
    input = torch.randn(batch_size, input_channels, depth, height, width, dtype=torch.double, requires_grad=True)
    out = conv_op(input)
    gO = torch.rand_like(out)
    out.backward(gO)
    print(conv_op.weight.grad)

import torch

input_channels = 3
output_channels = 3
batch_size = 2
depth=3
height = 5
width = 5
kernel = 1
stride = 1
with torch.backends.mkldnn.flags(enabled=False):
    conv_op = torch.nn.Conv3d(
    input_channels,
    output_channels,
    kernel,
    bias=False,  # No bias
    ).to(dtype=torch.double)
    input = torch.randn(batch_size, input_channels, depth, height, width, dtype=torch.double, requires_grad=True)
    out = conv_op(input)
    gO = torch.rand_like(out)
    out.backward(gO)
    print(conv_op.weight.grad)