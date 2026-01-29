# torch.rand(32, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nonzero(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a tensor with size (32,4) and strides (4,1)
    size = (32, 4)
    strides = (4, 1)
    x = torch.empty_strided(size, strides, dtype=torch.float32)
    x.copy_(torch.rand(size))  # Initialize with random values
    return x

