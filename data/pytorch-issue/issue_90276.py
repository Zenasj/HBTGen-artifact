# torch.rand(B=1, C=1, H=1, W=3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 3)

