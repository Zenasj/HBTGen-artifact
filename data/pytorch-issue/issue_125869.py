# torch.rand(100, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Mimic the behavior of the custom operator from the issue's example
        return torch.randn(2, 3, 4, 5)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(100, dtype=torch.int64)

