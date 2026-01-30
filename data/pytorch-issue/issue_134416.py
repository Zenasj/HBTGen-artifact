import torch.nn as nn

print(torch.__version__)

import torch

import torch
print(torch.__version__)
import torch.nn.functional as F

weight_cpu = torch.randn(1, 4, 10, device="cpu")
weight_mps = weight_cpu.detach().clone().to("mps")

nc = 65536 # OK
nc = 66000 # NotImplementedError: Output channels > 65536 not supported at the MPS device.
x_cpu = torch.randn(1, 4, nc, device="cpu")
x_mps = x_cpu.detach().clone().to("mps")

y_cpu = F.conv1d(x_cpu, weight_cpu)
y_mps = F.conv1d(x_mps, weight_mps)

print(y_cpu)
print(y_mps)
# Outputs:
# 2.6.0.dev20241212
# tensor([[[ 1.7427,  6.5344, -0.4782,  ...,  6.6598, -8.1508,  5.4256]]])
# tensor([[[ 1.7427,  6.5344, -0.4782,  ...,  6.6598, -8.1508,  5.4256]]],
#        device='mps:0')

import platform
platform.mac_ver()
# My output: ('15.2', ('', '', ''), 'arm64')