# torch.rand(2, 2, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()  # First submodule outputs input as-is
        
    def forward(self, x):
        # Second "model" adds an imaginary component of 1 to all elements
        modified = x + torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        return x, modified  # Returns tuple of original and modified tensors for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.complex64)  # Matches input shape from example 2

