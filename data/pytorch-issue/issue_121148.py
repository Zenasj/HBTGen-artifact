# Input is a tuple (torch.rand(()), torch.rand(6, 0, dtype=torch.float16))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        a, b = inputs
        return torch.inner(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(())
    b = torch.rand(6, 0, dtype=torch.float16)
    return (a, b)

