# torch.rand(3)
import torch
from contextlib import nullcontext
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, z=None):
        with nullcontext():
            with nullcontext():
                if z is None:
                    y = x ** 2
                else:
                    y = x ** 3
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

