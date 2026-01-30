import torch.nn as nn

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, x):
        return self.weight + x

class MyParametrization(nn.Module):
    def forward(self, X):
        return -X

@torch.compile(backend="eager")
def f():
    m = MyModule()
    register_parametrization(m, 'weight', MyParametrization())
    output = m(torch.randn(3))
    return output

output = f()
print(output)