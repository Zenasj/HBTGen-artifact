# torch.rand(2, 8, 2, 2) repeated 3 times in a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent  # stores exponent as a parameter

    def forward(self, inputs):
        # Reproduces the error scenario using keyword args with foreach_pow
        return torch._foreach_pow(self=inputs, exponent=self.exponent)

def my_model_function():
    # Initialize with exponent value from the issue's minified example
    return MyModel(exponent=2.7)

def GetInput():
    # Generate input tuple matching the issue's (t, t, t) structure
    t = torch.rand(2, 8, 2, 2)
    return (t, t, t)

