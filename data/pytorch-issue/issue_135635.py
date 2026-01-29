# torch.rand(4, 15, dtype=torch.float32)  # Inferred input shape from example context
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply view(-1) and multiplication to replicate test scenario
        return x.view(-1) * 2

def my_model_function():
    # Returns model instance for testing symbolic shape logic
    return MyModel()

def GetInput():
    # Generates input matching the view(-1) test case
    return torch.rand(4, 15, dtype=torch.float32)

