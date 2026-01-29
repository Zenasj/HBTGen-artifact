# torch.randint(256, size=(1024, 2048), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x << 4  # Reproduces the bitwise shift operation causing the error

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(256, size=(1024, 2048), dtype=torch.int64)

