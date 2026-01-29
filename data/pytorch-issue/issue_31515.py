# torch.rand(2, 0, dtype=torch.float32)  # Input shape with zero-length dimension
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return x.cumsum(dim=-1).sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(2, 0, requires_grad=True)

