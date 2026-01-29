# torch.rand(B, 3, 64, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.add(x, 1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 3, 64, 64, dtype=torch.float32)

