# torch.rand(2), torch.rand(3)  # Input is a tuple of two tensors with shapes (2,) and (3,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # y is unused, replicating the original function's behavior
        return x * 2

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2), torch.rand(3))  # Returns a tuple of tensors matching the expected input

