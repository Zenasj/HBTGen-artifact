# torch.rand(2, 2), torch.rand(2, 2)  # Input is a tuple of two 2x2 tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x @ y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 2), torch.rand(2, 2))

