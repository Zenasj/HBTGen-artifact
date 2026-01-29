# torch.rand(2, 3, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.round(x, decimals=3)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float16)

