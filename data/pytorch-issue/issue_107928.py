# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class InplaceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, other):
        ctx.mark_dirty(x)
        return x.mul_(2)

    @staticmethod
    def backward(ctx, grad):
        return grad, None  # Returns undefined gradient for second input

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.], requires_grad=True))
        self.b = nn.Parameter(torch.tensor([1.], requires_grad=True))

    def forward(self, x):
        c = self.a.clone()
        v = c[:]  # Create a view of the cloned tensor
        out = InplaceFunc.apply(v, self.b)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

