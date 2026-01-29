# torch.rand(100, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.var()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, dtype=torch.complex64)

