# torch.rand(10, dtype=torch.int64), torch.rand(10, dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x
        return (a > 1) & (b > 1)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.ones(10, dtype=torch.int64)
    b = torch.ones(10, dtype=torch.uint8)
    return (a, b)

