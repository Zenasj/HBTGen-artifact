# torch.rand(50, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Problematic .data assignment that fails under functionalization
        x_slice = x.data[:5]
        x.data = x_slice  # This line triggers the reported issue
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50, dtype=torch.float32)

