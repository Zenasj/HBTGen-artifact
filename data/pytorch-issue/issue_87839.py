# torch.rand(B, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)  # Matches the model in the issue's code example
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape (batch, 32)
    return torch.rand(2, 32)  # Batch size 2 as a minimal example

