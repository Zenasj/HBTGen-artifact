# torch.rand(B, 5, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)  # Matches input shape from example
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Initialize model with default weights
    return MyModel()

def GetInput():
    # Reproduces input shape from issue's example
    return torch.rand(5, 3)

