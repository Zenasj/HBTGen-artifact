# torch.rand(B, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple linear regression model matching the input shape (1 feature)
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Initialize model with default PyTorch weights
    return MyModel()

def GetInput():
    # Generate random input matching (Batch, Feature) shape expected by the model
    B = 5  # Example batch size (can be any positive integer)
    return torch.rand(B, 1, dtype=torch.float32)

