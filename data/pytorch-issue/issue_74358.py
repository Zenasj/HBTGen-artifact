# torch.rand(4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift  # Shift value as a parameter (float allowed)

    def forward(self, x):
        return x / (2 ** self.shift)  # Implements PyTorch's non-bitwise shift behavior

def my_model_function():
    # Initialize with shift=2 as shown in the issue example
    return MyModel(shift=2.0)

def GetInput():
    # Returns a 1D tensor matching the example's input shape
    return torch.rand(4, dtype=torch.float32)

