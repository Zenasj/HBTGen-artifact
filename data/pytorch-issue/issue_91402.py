# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Apply a Bessel function requiring vmap support (e.g., bessel_j0)
        return torch.special.bessel_j0(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 2, 3, 4, 5
    return torch.rand(B, C, H, W, dtype=torch.float32)

