# torch.rand(B, C, dtype=torch.float32)  # Inferred from example input torch.randn(13,14)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        if x.is_contiguous():
            return x
        else:
            return torch.zeros_like(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(13, 14)  # Matches the test input from the issue's example

