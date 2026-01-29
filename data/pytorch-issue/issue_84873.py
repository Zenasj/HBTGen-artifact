# torch.rand(3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def f(self, x):
        return x
    
    def forward(self, x):
        return self.f(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)

