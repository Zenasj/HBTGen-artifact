# torch.rand(3, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

class MyParametrization(nn.Module):
    def forward(self, X):
        return -X

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, x):
        return self.weight + x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = MyModule()
        register_parametrization(self.module, 'weight', MyParametrization())

    def forward(self, x):
        return self.module(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, dtype=torch.float32)

