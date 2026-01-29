# torch.rand(5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x, n=1):
        return x + n

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5)

