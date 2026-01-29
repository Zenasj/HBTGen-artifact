# torch.rand(1, 3, 1, 1, dtype=torch.float32)  # Inferred input shape (batch, channels, height, width)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Create a tensor using the default device (critical for demonstrating threading issue)
        default_tensor = torch.ones_like(x) * 2  # Uses current default device
        return x + default_tensor

def my_model_function():
    # Returns an instance of MyModel with no special initialization required
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape
    return torch.rand(1, 3, 1, 1, dtype=torch.float32)

