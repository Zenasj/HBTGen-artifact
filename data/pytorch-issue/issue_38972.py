# torch.rand(B, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example model structure based on inferred input shape (5 features)
        self.linear = nn.Linear(5, 10)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Initialize with default weights
    return MyModel()

def GetInput():
    # Generate random input matching the model's expected shape (B=1, 5 features)
    return torch.rand(1, 5, dtype=torch.float32)

