import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.randn(32, 3, 16, 16)
weight = torch.randn(3, 3, 4, 4, requires_grad=True)

@torch.compile(backend='aot_eager')
def f(x, weight):
    return F.conv2d(x, weight, stride=(1, 1))

f(x, weight)