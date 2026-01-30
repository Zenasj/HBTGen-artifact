import torch

y = torch.tensor(3)

class MulY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.y = y
        return x * y

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.y

@torch.compile(backend="eager", fullgraph=True)
def f(x):
    return MulY.apply(x)

x = torch.tensor(2., requires_grad=True)
out = f(x)
expected = MulY.apply(x)
assert torch.allclose(out, expected)