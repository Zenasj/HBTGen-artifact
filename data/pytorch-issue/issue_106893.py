py
import torch
from typing import *
from torch.autograd.function import once_differentiable


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return grad * 0.5

x = torch.randn(3, requires_grad=True)
def f(x):
    return ScaleGradient.apply(x)
output = torch.compile(f, backend='eager', fullgraph=True)(x)