py
import torch

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x.sin()
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.x.cos()

@torch.compile(backend='aot_eager')
def f(x):
    return Foo.apply(x)

x = torch.randn([], requires_grad=True)
f(x)