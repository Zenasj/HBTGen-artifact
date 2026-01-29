# torch.rand(B, 2, 2, dtype=torch.float64)  # Inferred input shape: batch of 2x2 matrices
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Applies torch.linalg.inv to invert the input matrix, which must be a square matrix (n x n)
        return torch.linalg.inv(x)

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns a random 2x2 tensor (valid for inversion)
    return torch.rand(2, 2, dtype=torch.float64)

