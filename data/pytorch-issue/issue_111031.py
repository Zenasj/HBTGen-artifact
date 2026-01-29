# torch.rand(2, dtype=torch.float32)
import torch
from torch.autograd import Function
import torch.nn as nn

class MyModel(nn.Module):
    class Foo(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.x0 = x.size(0)
            return x * 2

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out * ctx.x0

    def forward(self, x):
        return MyModel.Foo.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, requires_grad=True)

