# torch.rand(1, 1, 1, dtype=torch.float32)  # Inferred input shape from the issue's example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize y as a buffer to match the original example's setup
        self.register_buffer('y', torch.randn(1, 1))  # Shape from the issue's y=torch.randn(1,1)
        
    def forward(self, x):
        return torch.matmul(x, self.y)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the 3D input that triggers the symbolic shape error in the issue
    return torch.randn(1, 1, 1)

