# torch.rand(2, 3, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.sgn(x)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.randn(2, 3, dtype=torch.complex128)
    input.requires_grad_(True)
    return input

