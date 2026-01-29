# torch.rand(5)  # Inferred input shape as a 1D tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.atleast_2d(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random 1D tensor as input
    return torch.rand(5)

