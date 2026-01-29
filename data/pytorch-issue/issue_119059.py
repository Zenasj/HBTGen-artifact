# torch.rand(4, 128, requires_grad=True)  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = UseNeedsInputGradFunction.apply

    def forward(self, x):
        return self.linear(x)

class UseNeedsInputGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0]:
            return grad_output
        return None

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, 128, requires_grad=True)

