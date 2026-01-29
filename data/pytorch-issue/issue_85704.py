# torch.randint(0, 10, (2, 2), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        b = True
        return x.square() | b  # Problematic bitwise OR with bool

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (2, 2), dtype=torch.int64)

