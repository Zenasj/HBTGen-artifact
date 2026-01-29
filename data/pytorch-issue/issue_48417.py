# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example parameter from the issue's addition operation (y = [2,3])
        self.y = nn.Parameter(torch.tensor([2.0, 3.0])) 

    def forward(self, x):
        # Replicates the operation z = x + y from the issue's example
        return x + self.y

def my_model_function():
    # Returns an instance of MyModel with default parameters
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape (B, 2)
    B = 1  # Default batch size (can be adjusted)
    return torch.rand(B, 2, dtype=torch.float32)

