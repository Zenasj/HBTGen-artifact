# torch.rand(B, 2, dtype=torch.float32)  # Input shape inferred from Linear(2,2) model in the issue's reproduction steps
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)  # Matches the Linear layer used in the issue's example

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generate random input matching the model's expected input shape (batch_size, 2)
    B = 1  # Arbitrary batch size (can be adjusted)
    return torch.rand(B, 2, dtype=torch.float32)

