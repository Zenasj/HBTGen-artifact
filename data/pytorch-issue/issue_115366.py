# torch.rand(100, 100, dtype=torch.float32, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(3.0))  # Matches the original affine function's "a" parameter
        self.b = nn.Parameter(torch.tensor(2.0))  # Matches the original affine function's "b" parameter
    
    def forward(self, x):
        return self.a * x + self.b  # Replicates the "a*x + b" computation

def my_model_function():
    # Returns a model instance with parameters initialized to match the original affine function
    return MyModel()

def GetInput():
    # Generates a random tensor matching the input shape and device used in the original reproducer
    return torch.randn(100, 100, device="cuda")

