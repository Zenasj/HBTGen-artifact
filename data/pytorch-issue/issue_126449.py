# torch.rand(5, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Apply constant_pad_nd with padding [2, 3] to 1D tensor
        return torch.constant_pad_nd(x, [2, 3])

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random bool tensor matching input shape (5,)
    return torch.randint(0, 2, (5,), dtype=torch.bool)

