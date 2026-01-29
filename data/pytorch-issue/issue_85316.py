# torch.rand((), dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return x * self.m(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((), dtype=torch.float64)

