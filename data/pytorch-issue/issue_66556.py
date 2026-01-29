# torch.rand((), dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x  # Identity model matching the OpInfo's lambda op

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.bfloat16)

