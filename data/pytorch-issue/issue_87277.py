# torch.rand(50, 9, 300, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.nn.functional.pad(x, (0, 0, 0, 31))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50, 9, 300, dtype=torch.float32)

