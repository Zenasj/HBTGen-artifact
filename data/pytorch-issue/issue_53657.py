# (torch.rand(1), torch.rand(1))  # Input is a tuple of two tensors
import torch
from torch import nn
from typing import Any

class MyModel(nn.Module):
    def forward(self, x: Any):
        if isinstance(x, tuple):
            a, b = x
            return a + b
        else:
            return x

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(1), torch.rand(1))

