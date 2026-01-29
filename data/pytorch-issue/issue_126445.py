# torch.rand(B, 2, 10)  # Input shape (batch, features)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10)])
    
    def forward(self, x):
        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    return torch.rand(B, 10)

