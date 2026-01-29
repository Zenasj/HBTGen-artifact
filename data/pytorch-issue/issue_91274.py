# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, b):
        return self.func(a=1, **{'b': b})
    
    def func(self, a, b):
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2)

