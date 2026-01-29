# torch.rand((), dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.complex64)

