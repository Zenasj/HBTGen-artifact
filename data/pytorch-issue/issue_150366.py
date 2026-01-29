# torch.rand(2, dtype=torch.complex64), torch.rand(2, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return torch.vdot(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(2, dtype=torch.complex64)
    b = torch.rand(2, dtype=torch.complex64)
    return (a, b)

