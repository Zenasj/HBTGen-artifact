# torch.rand(B, 10, dtype=torch.float32)  # Assuming input shape based on common use of vmap with linear layers
import torch
import torch.nn as nn
from torch.func import vmap  # Corrected import path for PyTorch 2.0+ as per user's version info

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Example layer using vmap-compatible operations

    def forward(self, x):
        # Apply vmap to vectorize the linear layer over the batch dimension
        return vmap(self.linear)(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching the expected shape (B, 10)
    return torch.rand(32, 10, dtype=torch.float32)

