# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x + x  # Reproduces the original issue's computation pattern

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)  # Matches the input shape expected by MyModel

