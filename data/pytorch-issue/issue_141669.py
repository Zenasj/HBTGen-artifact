# torch.rand(2, requires_grad=True) ← Add a comment line at the top with the inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.custom_fn = CustomFn.apply

    def forward(self, x):
        return self.custom_fn(x)

class CustomFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        # Perform an in-place operation
        return torch.exp_(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, requires_grad=True)

