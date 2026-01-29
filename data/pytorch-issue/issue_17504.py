# torch.randint(0, 10, (2,), dtype=torch.int32)  # Input shape: two integers (numerator, denominator)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x[0], x[1]
        return a / b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (2,), dtype=torch.int32)

