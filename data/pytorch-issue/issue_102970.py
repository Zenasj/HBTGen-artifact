py
import torch

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gx):
        return gx * 0.5

x = torch.randn([], requires_grad=True)

def f(x):
    return Foo.apply(x)

y = torch.compile(f)(x)
result, = torch.autograd.grad(y, x)
print(result)  # 1.0, but should be 0.5