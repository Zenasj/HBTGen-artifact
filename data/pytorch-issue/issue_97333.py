# torch.rand(4, 4, dtype=torch.float16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.fmod(x, 2.3)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((4,4), dtype=torch.float16)

