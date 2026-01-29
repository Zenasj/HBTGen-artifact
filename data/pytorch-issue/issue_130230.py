# torch.rand(18, 7, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.cdist(x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(18, 7)

