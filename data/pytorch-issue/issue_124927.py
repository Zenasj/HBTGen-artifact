# torch.rand(2, 3, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        if x.is_complex():
            return x + 1
        else:
            return x - 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, dtype=torch.complex64)

