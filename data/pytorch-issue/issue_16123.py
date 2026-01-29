# torch.rand(3, 2, dtype=torch.float)  # Input shape (batch=3, features=2)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])  # Example ModuleList of 3 layers

    def forward(self, x):
        for mod in self.mods:
            x = mod(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 2)  # Batch size 3, input features 2

