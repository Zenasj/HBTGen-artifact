import torch
from torch import autograd

class Foo(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Inputs/output should not be saved in ctx
        # But use ctx.save_for_backward()
        ctx.save_for_backward(x)
        x1 = x.clone()
        ctx.x1 = x1
        x2 = x1.clone()
        return x2

    @staticmethod
    def backward(ctx, gx):
        print(ctx.x1)
        print(ctx.saved_tensors)
        return gx.clone()

inp = torch.rand(10, requires_grad=True)

Foo.apply(inp).sum().backward()
print("Ok")

with torch.no_grad():
    Foo.apply(inp).sum()
print("Ok")