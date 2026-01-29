# torch.rand(1, 2, 1, 2, 1, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        v = x.select(2, 0)  # Selects 0th index along dimension 2 (third position)
        return v.add_(1)    # In-place addition triggers the compile error

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, 2, 1, 2)  # Matches the input shape in the original issue

