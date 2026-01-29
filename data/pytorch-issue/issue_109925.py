# torch.rand(2, 4, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        t2 = x.to(dtype=torch.bool)
        t3 = torch.cumsum(t2, dim=1)
        return t3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.full((2, 4), 1, dtype=torch.float)

