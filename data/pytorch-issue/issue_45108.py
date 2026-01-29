# torch.rand(5, dtype=torch.float32)  # Input shape is (5,)
import torch
import warnings
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        for _ in range(10):
            x = torch.nn.functional.softmax(x)
            warnings.warn("Test warning message", stacklevel=2)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5)

