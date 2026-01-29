# torch.rand(10, 100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        a = torch.sin(x)
        b = torch.cos(a)
        return b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 100, dtype=torch.float32)

