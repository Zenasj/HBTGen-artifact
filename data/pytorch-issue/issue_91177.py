# torch.rand(B, 3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply index_fill on dimension 1 with indices [0, 2] and value -1.0
        return x.index_fill(1, torch.tensor([0, 2], device=x.device), -1.0)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected shape (B, 3, 3)
    B = 4  # Example batch size, can be adjusted
    return torch.randn(B, 3, 3, dtype=torch.float32)

