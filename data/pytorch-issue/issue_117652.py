# (torch.rand(2), torch.rand(2)) ← inferred input shape (two tensors of shape (2,))
import torch
from torch import nn

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, out_grad):
        x, y = ctx.saved_tensors
        return out_grad * x, out_grad * y

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return MyFunc.apply(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2, requires_grad=True)
    y = torch.rand(2, requires_grad=True)
    return (x, y)

