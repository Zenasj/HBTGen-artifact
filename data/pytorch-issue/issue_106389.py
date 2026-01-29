# torch.randn((), dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin(), x.sin()
    
    @staticmethod
    def backward(ctx, grad0, grad1):
        x, = ctx.saved_tensors
        return grad * x.cos(), grad * x.cos()  # Matches original typo in the issue's code

class MyModel(nn.Module):
    def forward(self, x):
        return Foo.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((), requires_grad=True)

