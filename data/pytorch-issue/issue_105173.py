# torch.rand((), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        tmp = x.ceil()
        x.add_(10)
        return tmp

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros((), dtype=torch.int64)

