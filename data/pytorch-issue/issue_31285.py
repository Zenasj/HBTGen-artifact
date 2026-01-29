# torch.rand(B, 2, dtype=torch.float32)  # Input shape inferred from example tensor [1.0, 2.0]
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear layer to demonstrate model structure
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a simple model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape (B=1, features=2)
    return torch.rand(1, 2, dtype=torch.float32)

