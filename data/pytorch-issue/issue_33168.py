import torch.nn as nn

import torch

in_channels = 64
out_channels = 128
scale_factor = 8
batch_size = 8
length = 16

conv = torch.nn.ConvTranspose1d(
    in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor)
layer_norm = torch.nn.LayerNorm(out_channels)

input_ = torch.randn(batch_size, in_channels, length).contiguous()
input_ = conv(input_).contiguous()
input_ = layer_norm(input_.transpose(1, 2).contiguous()).contiguous()
input_.sum().backward()