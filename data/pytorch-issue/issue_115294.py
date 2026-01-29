# torch.rand(10, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        with torch.enable_grad():
            out = x + 1
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, requires_grad=True)

