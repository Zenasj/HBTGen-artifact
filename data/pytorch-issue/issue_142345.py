# torch.rand(4, dtype=torch.float32)  # Shape must be at least 4 elements to trigger the bug
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, y):
        return torch.asinh(y)

def my_model_function():
    return MyModel()

def GetInput():
    # Create input with shape >=4 and at least one element with negative value < -10000.0 and non-zero decimal
    x = torch.randn(4)  # Random values for first 3 elements
    x[-1] = -10000.1  # Set last element to trigger -inf in CPU Inductor
    return x

