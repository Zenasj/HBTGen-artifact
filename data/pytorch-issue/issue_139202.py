# torch.rand(2, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

