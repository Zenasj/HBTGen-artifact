# torch.rand(23, 20, 34, 15, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Apply slicing as in the original issue
        x = x[:,:-3, :-2]  # Slice second and third dimensions
        # Use reshape instead of view to handle non-contiguous memory
        return x.reshape(32, 15, 23, 17)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(23, 20, 34, 15)

