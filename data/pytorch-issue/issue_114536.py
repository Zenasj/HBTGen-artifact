# torch.rand(3, dtype=torch.double, device='cuda', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x ** 2.0

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.double, device='cuda', requires_grad=True)

