# torch.rand(1, dtype=torch.float32)  # Inferred input shape from the issue

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.foo = Foo.apply

    def forward(self, x):
        return self.foo(x)

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, gx):
        x, = ctx.saved_tensors
        return gx * 0.5

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn([], requires_grad=True)

