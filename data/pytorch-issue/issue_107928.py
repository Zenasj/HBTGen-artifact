import torch

a = torch.tensor([1.], requires_grad=True)
c = a.clone()
v = c[:]
b = torch.tensor(1., requires_grad=True)

class InplaceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        ctx.mark_dirty(x)
        return x.mul_(2)

    @staticmethod
    def backward(ctx, grad):
        return grad, None

out = InplaceFunc.apply(v, b)

torch.autograd.grad(out, inputs=(a, b))