# torch.rand(10, dtype=torch.float32)  # Input is a 1D tensor defining the desired output shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, high):
        super().__init__()
        self.high = high

    def forward(self, x):
        # Generate integers using the input tensor's shape (avoids 'out' parameter issue)
        return torch.randint(self.high, x.shape)

def my_model_function():
    # Initialize with high=17 as in the original repro
    return MyModel(high=17)

def GetInput():
    # Return a tensor whose shape defines the desired output size (e.g., 10 or 12 elements)
    return torch.empty(10)  # Matches first test case in the repro

