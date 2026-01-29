# torch.rand(B, 8, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.rand(8))  # Initialize a parameter for comparison

    def forward(self, x):
        return torch.maximum(x, self.y)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Default batch size (can be adjusted)
    return torch.rand(B, 8)

