# torch.rand(1, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.cat(2 * [x], dim=0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, dtype=torch.float32)

