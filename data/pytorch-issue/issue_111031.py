import torch
from torch.autograd import Function

class Foo(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x0 = x.size(0)
        return x * 2

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.x0


@torch.compile(backend="eager", fullgraph=True, dynamic=True)
def foo(x):
    return Foo.apply(x)

foo(torch.randn(2, requires_grad=True))