# torch.rand(354298880, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute norms in different dtypes and return their differences
        norm_float = x.norm()
        norm_double = x.double().norm()
        norm_half = x.half().norm()
        # Return the three norms for comparison
        return (norm_float, norm_double, norm_half)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(354298880, dtype=torch.float32)

