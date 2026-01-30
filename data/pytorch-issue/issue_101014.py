import torch.nn as nn

import math
import torch
import torch.nn.functional as F

kernel = torch.randn(1, 1, 4)

def func(x):
    padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
    out = F.conv1d(x, kernel, padding=padding, stride=2)
    return out

opt_func = torch.compile(func, dynamic=True)

x = torch.randn(1, 1, 175)
opt_func(x)  # passes
x = torch.randn(1, 1, 249)
opt_func(x)  # crashes

import math
import torch
import torch.nn.functional as F

kernel = torch.randn(1, 1, 4)


def func(x):
    parity = 0 if x.shape[-1] % 2 == 0 else 1  # new line
    padding = math.ceil((kernel.shape[-1] + parity) / 2) - 1
    out = F.conv1d(x, kernel, padding=padding, stride=2)
    return out


opt_func = torch.compile(func, dynamic=True)

x = torch.randn(1, 1, 175)
opt_func(x)  # passes
x = torch.randn(1, 1, 249)
opt_func(x)  # does not crash anymore