# torch.rand(1, 2, 3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert tensor size to Python int to avoid tracing errors
        split_size = int(x.size(1))
        return torch.split(x, split_size)  # Split along dimension 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2, 3, 4)

