# torch.rand(B, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Simple linear layer to process scalar inputs

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape (B=1, 1 feature)
    return torch.rand(1, 1, dtype=torch.float32)  # Matches the input shape comment

