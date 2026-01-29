# torch.rand(2, 3, dtype=torch.float)  # Input is a tensor split into x and y
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input):
        x, y = input.chunk(2, dim=0)  # Split input into two parts
        del x, y  # This line triggers TorchScript's "del with multiple operands" error
        return torch.tensor(3, dtype=torch.float)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float)  # Input tensor to be split into x and y

