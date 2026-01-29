# torch.rand(1, 1, 1, 2, dtype=torch.float32)  # Inferred input shape from example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.squeeze(x)  # Reproduces the working case from the issue's example

def my_model_function():
    return MyModel()  # Returns a model instance without requiring extra initialization

def GetInput():
    # Returns a 4D tensor matching the model's expected input shape
    return torch.rand(1, 1, 1, 2, dtype=torch.float32)

