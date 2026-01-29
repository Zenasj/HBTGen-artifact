# torch.rand(1, dtype=torch.float32)

import torch
from torch import nn

class Anything(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.new_ones((5,))

    @staticmethod
    def backward(ctx, grad_outputs):
        # Preserve the original backward logic for demonstration
        print("Is gradient w.r.t. sum of b contiguous? ", grad_outputs.is_contiguous())
        return grad_outputs.new_zeros((1,))

class MyModel(nn.Module):
    def forward(self, x):
        b = Anything.apply(x)
        return b.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, requires_grad=True)

