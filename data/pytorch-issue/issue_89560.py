# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()  # Default 1D PReLU with single weight
        
    def forward(self, x):
        return self.prelu(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4)  # Matches 1D input shape (batch=4, no channels/spatial dims)

