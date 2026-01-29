# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return MyOp.apply(x, x[0], x)

class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        return a.svd()

    @staticmethod
    def backward(ctx, a, b, c):
        return a.svd()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5, dtype=torch.float32).requires_grad_(True)

