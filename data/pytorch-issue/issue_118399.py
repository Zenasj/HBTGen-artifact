import torch

foo = []

class MulY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x * 3

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out.stride(0) == 1:
            foo.append(grad_out)
            return grad_out * 2
        return grad_out * 3

@torch.compile(backend="eager")
def f(x):
    return MulY.apply(x)

x = torch.tensor(2., requires_grad=True)
expected = MulY.apply(x)
out = f(x)
assert torch.allclose(out, expected)