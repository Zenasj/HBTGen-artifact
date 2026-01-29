# torch.rand(3, 3, dtype=torch.float32), torch.rand(3, 3, dtype=torch.float32)  # as a tuple of two inputs
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.mm(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    dtype = torch.float32
    x = torch.randn(3, 3, dtype=dtype)
    y = torch.randn(3, 3, dtype=dtype)
    return (x, y)

