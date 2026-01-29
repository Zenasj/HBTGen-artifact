# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)  # Matches input dimension from GetInput()
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Initialize model with default parameters
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    return torch.rand(1, 3, dtype=torch.float32)  # Batch size 1, 3 features

