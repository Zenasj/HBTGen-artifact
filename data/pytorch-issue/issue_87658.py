# torch.rand(B, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        n = 32  # As in the original example
        a = x * 1.0 / (n - 1)
        b = x * (1.0 / (n - 1))
        # Return boolean indicating if all elements are equal (due to float32 precision differences)
        return torch.all(a == b)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (1, 1) with float32
    return torch.rand(1, 1, dtype=torch.float32)

