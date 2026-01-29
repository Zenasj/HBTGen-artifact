# torch.rand(2, dtype=torch.double)
import torch
from torch import nn, autograd

class BadCustomFunction(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        interm = inp * 2
        ctx.foo = interm
        res = interm ** 2
        return res

    @staticmethod
    def backward(ctx, gres):
        grad = 2 * 2 * ctx.foo * gres
        return grad

class MyModel(nn.Module):
    def forward(self, x):
        return BadCustomFunction.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.double, requires_grad=True)

