# torch.rand(1, 1, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x ** x  # Core operation causing stack traces in profiler

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    return torch.rand(1, 1, dtype=torch.float32, requires_grad=True)  # Matches profiler test requirements

