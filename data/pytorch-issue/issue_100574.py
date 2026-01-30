import torch

from torch import randn
from torch.autograd import Function
from torch.autograd.forward_ad import dual_level
from torch.autograd.forward_ad import make_dual


class TestFunc1(Function):
    @staticmethod
    def forward(ctx, x):
        return 1, x

    @staticmethod
    def backward(ctx, dy, dz):
        return dz

    @staticmethod
    def jvp(ctx, dz):
        return None, dz


x = randn(5, requires_grad=True)

# this works

z2 = TestFunc1.apply(x)[1].sum().backward()
assert x.grad is not None


# this breaks

dx = randn(5)
with dual_level():
    x2 = make_dual(x, dx)
    z2 = TestFunc1.apply(x2)  # raises RuntimeError: bad optional access