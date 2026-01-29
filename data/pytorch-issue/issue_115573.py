# (torch.rand(10), torch.rand(10))  # Input shape inferred from example
import torch
from torch import nn

class MyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.sin(x), torch.sin(y)
    
    @staticmethod
    def backward(ctx, gO_x, gO_y):
        x, y = ctx.saved_tensors
        return gO_x * torch.cos(x), gO_y * torch.cos(y)

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return MyFn.apply(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(10), torch.rand(10))

