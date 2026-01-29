# torch.rand(2), torch.rand(2)  # Example input shapes (y is used, x is unused)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # Unpack the tuple input (x is unused)
        return y + 2

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2), torch.rand(2))  # Returns a tuple of two tensors

