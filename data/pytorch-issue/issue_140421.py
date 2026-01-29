# torch.rand(64, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nonzero(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(64, 10)

