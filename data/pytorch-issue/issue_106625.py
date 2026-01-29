# torch.rand(4, 10), torch.rand(4, 10)  # Input shapes for x and y
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return (x * x) * y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(4, 10), torch.rand(4, 10))

