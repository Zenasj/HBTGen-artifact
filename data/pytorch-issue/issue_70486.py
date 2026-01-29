# Input is a tuple of three tensors: (torch.rand(5,5, dtype=torch.complex128), torch.rand(5,5, dtype=torch.complex128), torch.rand(1,5, dtype=torch.complex128))

import torch
from torch import nn

class AddCmul(nn.Module):
    def forward(self, a, b, c):
        return torch.addcmul(a, b, c)

class AddCdiv(nn.Module):
    def forward(self, a, b, c):
        return torch.addcdiv(a, b, c)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.addcmul = AddCmul()
        self.addcdiv = AddCdiv()

    def forward(self, inputs):
        a, b, c = inputs
        return (self.addcmul(a, b, c), self.addcdiv(a, b, c))

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(5, 5, dtype=torch.complex128)
    b = torch.rand(5, 5, dtype=torch.complex128)
    c = torch.rand(1, 5, dtype=torch.complex128)
    return (a, b, c)

