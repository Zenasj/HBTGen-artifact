import torch.nn as nn

import torch
import torch.nn.functional as F

@torch.compile(dynamic=True)
def f(x):
    return F.interpolate(x, scale_factor=1 / 300, mode="linear")

f(torch.randn(1, 8, 396 * 300)).shape  # torch.Size([1, 8, 395]) -> wrong shape, should be (1, 8, 396)

@torch.compile(dynamic=True)
def f(x):
    return F.interpolate(x, size=(x.shape[-1] // 300,), mode="linear")