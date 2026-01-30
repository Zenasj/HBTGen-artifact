import torch.nn as nn
import numpy as np

import torch
from classy_vision.models import build_model

torch.manual_seed(123)
dtype = torch.float16
model = build_model({"name": "resnext101_32x4d"}).eval().cuda()

# By default BatchNorm weights are set to 0. This prevents that.
model._initialize_weights(False)
model = model.to(dtype)
device = torch.device('cuda')
inp = torch.randn(2, 3, 50, 50, device=torch.device('cuda'), dtype=dtype)
out1 = model(inp)
inp = inp.contiguous(memory_format=torch.channels_last)
out2 = model(inp)

print((out1 - out2).abs().mean().item())

# Using float16 tolerances from torch/testing/_core.py
print(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3))

0.0718994140625
False

import torch
from classy_vision.models import build_model

torch.manual_seed(123)
dtype = torch.float16
model = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
model = model.eval().cuda()
model = model.to(dtype)
device = torch.device('cuda')
inp = torch.randn(2, 64, 50, 50, device=torch.device('cuda'), dtype=dtype)
out1 = model(inp)
inp = inp.contiguous(memory_format=torch.channels_last)
out2 = model(inp)

print((out1 - out2).abs().mean().item())
print(torch.allclose(out1, out2, rtol=1e-3, atol=1e-3))

1.12890625
False