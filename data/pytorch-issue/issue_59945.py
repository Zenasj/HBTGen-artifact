# torch.rand(1, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.conj()  # Demonstrates the conjugate operation causing numpy interoperability issues

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.complex64)  # Matches the input expected by MyModel

