# (torch.rand(3), torch.rand(3))  # Example input: tuple of two tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # Inputs are a tuple of two tensors
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tuple of two random tensors matching the expected input format
    return (torch.rand(3), torch.rand(3))

