# torch.rand(1, 1, 1, 128, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.cumsum(dim=-1)  # Test cumsum along the last dimension (size 128)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(1, 1, 1, 128, dtype=torch.bool)

