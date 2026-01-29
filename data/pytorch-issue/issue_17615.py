# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Create Long tensor and perform assignment (truncates values)
        y_long = torch.zeros_like(x, dtype=torch.long)
        y_long[:] = x[:]  # Truncates float values to integers
        y_long_float = y_long.to(torch.float)
        
        # Compute difference between original and truncated values
        difference = x - y_long_float
        return difference

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 1D tensor of 10 elements matching the example's input shape
    return torch.rand(10)

