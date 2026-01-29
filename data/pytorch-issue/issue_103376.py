# torch.rand(B, 1, dtype=torch.float32)  # Inferred input shape: scalar parameter, input can be any shape (e.g., (B, 1))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(5.0))  # Initialize a scalar Parameter

    def forward(self, x):
        # Apply in-place operation to demonstrate type preservation (fixed by PR)
        self.param.mul_(2.0)
        return x * self.param  # Example usage of the modified Parameter

def my_model_function():
    # Returns an instance with the Parameter initialized above
    return MyModel()

def GetInput():
    # Returns a random tensor compatible with the model's forward method
    return torch.rand(1, 1, dtype=torch.float32)  # (batch=1, scalar)

