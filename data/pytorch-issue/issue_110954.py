# torch.rand(1000, dtype=torch.float32)  # Input is a 1D tensor of 1000 elements
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, scalar):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        tensors = list(x)  # Convert tensor to list of 0-dim tensors
        return torch._foreach_add(tensors, self.scalar)  # Returns new list of tensors

def my_model_function():
    # Uses the scalar value from the original test code
    return MyModel(1024.1024)

def GetInput():
    # Generates a list-compatible input tensor of shape (1000,)
    return torch.rand(1000, dtype=torch.float32)

