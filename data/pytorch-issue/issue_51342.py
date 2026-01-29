# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape  # Target shape for reshape
    
    def forward(self, a):
        # Reshape input tensor to the predefined shape and return sum
        b = a.reshape(self.shape)
        return b + b

def my_model_function():
    # Initialize with target shape (3,2) as per example's context
    return MyModel(shape=(3, 2))

def GetInput():
    # Input tensor with 6 elements (matching the product of target shape 3x2)
    return torch.rand(2, 3)

