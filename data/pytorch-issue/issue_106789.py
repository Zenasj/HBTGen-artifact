# torch.rand(4, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.prod(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, requires_grad=True)

