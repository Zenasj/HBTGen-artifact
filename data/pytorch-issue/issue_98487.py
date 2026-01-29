# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            return x + 2
        else:
            return x + 3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10)

