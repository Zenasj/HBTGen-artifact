import torch.nn as nn

import torch
import torch._dynamo
import torch.nn.functional as F

@torch.compile
def g(x):
    return x + 1

@torch.compile
def f(x, weight, y):
    result = F.linear(x, weight, bias=None)
    torch._dynamo.graph_break()
    result += y
    return result

x = g(torch.randn(4, 155, 4096, requires_grad=True))
f(x, torch.randn(12288, 4096), torch.randn(4, 155, 12288, requires_grad=True))