# torch.zeros(309237982, 2, 5, dtype=torch.int8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x  # Pass-through to exercise tensor handling

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(309237982, 2, 5, dtype=torch.int8)

