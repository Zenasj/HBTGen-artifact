# torch.rand(10, 1, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x  # Identity model to demonstrate tensor generation

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor of shape (10,1) to simulate input for MyModel
    return torch.arange(10).unsqueeze(1).float()

