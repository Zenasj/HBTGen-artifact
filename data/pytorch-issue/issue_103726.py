# torch.rand(1, dtype=torch.float, requires_grad=True)
import torch
from torch import nn

class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        intermediate = x.exp()
        # Save a tensor with requires_grad=True to trigger the issue
        ctx.save_for_backward(intermediate.clone().detach_().requires_grad_(True))
        return x.exp()

    @staticmethod
    def backward(ctx, grad_out):
        intermediate, = ctx.saved_tensors
        return grad_out * intermediate

class MyModel(nn.Module):
    def forward(self, x):
        return Func.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float, requires_grad=True)

