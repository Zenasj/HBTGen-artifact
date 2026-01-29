# (torch.rand((), dtype=torch.float32), torch.randint(10, ())) ← inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
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

    def forward(self, inputs):
        x, y = inputs
        return self.A.apply(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand((), dtype=torch.float32)
    y = torch.randint(10, ())
    return (x, y)

