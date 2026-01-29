# torch.rand(1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Perform repeat and repeat_interleave operations to test dynamic shape handling
        b = x.repeat(10, 1, 1)
        c = x.repeat_interleave(repeats=10, dim=0)
        return b, c  # Return both outputs for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, dtype=torch.float32)

