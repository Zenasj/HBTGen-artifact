# torch.rand(1, dtype=torch.bool), torch.rand(1, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.tensor((True,), dtype=torch.bool)
    b = torch.tensor((1 + 1j,), dtype=torch.complex128)
    return (a, b)

