import torch
from torch import nn

class Base(nn.Module):
    __constants__ = ["x"]
    x: float

    def __init__(self):
        super().__init__()
        self.x = 5.0

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(64)
        self.base = Base()

    def forward(self, x):
        # Problem 1: Attempt to modify constant 'momentum' of BatchNorm
        self.bn.momentum = 1.0
        # Problem 2: Attempt to modify constant 'x' of Base
        self.base.x = 6.0
        bn_out = self.bn(x)
        return bn_out + self.base.x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, 32, 32, dtype=torch.float32)

