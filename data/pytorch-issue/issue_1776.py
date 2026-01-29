# torch.rand(1, 2, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class exampleFct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        with torch.enable_grad():
            y = x ** 2  # Re-compute forward pass inside backward
            return torch.autograd.grad(y, x, dy)[0]  # Extract scalar from tuple

class MyModel(nn.Module):
    def forward(self, x):
        return exampleFct.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, 1, dtype=torch.float32)

