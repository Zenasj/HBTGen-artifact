py
import torch

class A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        out = y.clone()
        ctx.mark_non_differentiable(out)
        return x.clone(), out

    @staticmethod
    def jvp(ctx, x_tangent, y_tangent):
        ctx.set_materialize_grads(True)
        return x_tangent, None

x = torch.tensor(2.)
x_tangent = torch.tensor(1.)
y = torch.tensor(3)

with torch.autograd.forward_ad.dual_level():
    x_dual = torch.autograd.forward_ad.make_dual(x, x_tangent)
    result = A.apply(x_dual, y)