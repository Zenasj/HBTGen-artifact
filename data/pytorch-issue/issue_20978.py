# torch.rand(B, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(x)

def my_model_function():
    # Initialize with dim=100 as in the original issue's Test(100)
    return MyModel(100)

def GetInput():
    # Generate input tensor matching the LayerNorm's expected last dimension (100)
    return torch.rand(5, 100, dtype=torch.float32)

