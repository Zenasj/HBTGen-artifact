# torch.rand(10, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * 2
        x = x / 3
        return x.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32, requires_grad=True)

