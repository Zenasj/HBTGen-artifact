import torch.nn as nn

import torch

conv_stride = 2
num_channels = 1
wave_to_conv = torch.rand((1, 1, 5024))
weights = torch.rand(1, 1, 25)
conv = torch.nn.functional.conv1d(wave_to_conv, weights, stride=conv_stride)