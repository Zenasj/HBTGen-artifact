# torch.rand(B, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Reproduces the dsplit issue with dynamic shapes
        return torch.dsplit(x, [1, 2, 3])

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the issue's test case
    return torch.randn(4, 4, 4)

