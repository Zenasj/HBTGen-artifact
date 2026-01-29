# torch.rand(10, 8, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        transposed = x.t()
        sliced = transposed[::2, ::2]
        return sliced.clone()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 8, dtype=torch.complex64)

