# torch.randn(8, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.bernoulli(x, 0.5)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(8, 32)

