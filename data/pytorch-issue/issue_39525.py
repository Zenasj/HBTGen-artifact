# torch.rand(3)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.to(device=None, copy=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)

