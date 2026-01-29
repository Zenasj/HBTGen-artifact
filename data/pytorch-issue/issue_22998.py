# torch.rand(1, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn
from torch.autograd import Function

class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, g):
        return g

class MyModel(nn.Module):
    def forward(self, x):
        return MyFunction.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, requires_grad=True)

