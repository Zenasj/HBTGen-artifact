# torch.rand(128, 1, requires_grad=True) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.toy_fn = toy_fn()

    def forward(self, x):
        return self.toy_fn.apply(x)

class toy_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        with torch.no_grad():
            torch.exp(inputs, out=inputs)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(128, 1, requires_grad=True)

