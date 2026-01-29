# torch.rand(1, 3, dtype=torch.float32)
import torch
import warnings
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        warnings.warn("moo")
        return x + x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3)

