# torch.rand(100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute first buffer using indices (i*i + i)
        indices = torch.arange(x.size(0), device=x.device)
        first = indices * indices + indices
        # Compute second buffer using first buffer values (first^2 - first)
        second = first * first - first
        return second

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, dtype=torch.float32)

