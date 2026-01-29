# torch.rand(B, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)  # Matches the Linear(32, 32) model in the issue
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    # Returns a random input tensor of shape (B, 32) compatible with MyModel
    return torch.rand(1, 32)  # B=1 as minimal example, dtype=torch.float32 by default

