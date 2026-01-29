# torch.rand(8, 8, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset  # Matches the example's offset (0)

    def forward(self, input_tensor):
        return torch.diag(input_tensor, self.offset)

def my_model_function():
    # Initialize with offset 0 as in the original example
    return MyModel(offset=0)

def GetInput():
    # Generate 8x8 tensor matching the input shape from the issue's example
    return torch.rand((8,8), dtype=torch.float)

