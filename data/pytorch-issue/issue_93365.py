# torch.rand(1, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        v7 = torch.cat([x, x], dim=0)
        v1 = torch.mul(v7, v7)
        return v7, v1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, dtype=torch.float32)

