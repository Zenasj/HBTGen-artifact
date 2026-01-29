# torch.rand(2, 3, dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply the problematic MPS operation
        x[x == 2] = -1
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random integer tensor with possible 2s to trigger the bug
    return torch.randint(0, 4, (2, 3), dtype=torch.int64)

