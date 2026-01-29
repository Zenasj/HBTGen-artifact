# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x + x  # Matches the behavior of the test case where x is added to itself

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    return torch.randn(3)  # Matches the test input shape (3,) from the global variable example

